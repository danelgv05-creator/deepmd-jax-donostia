import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, lax
import flax.linen as nn
from .utils import *
from jax.sharding import PartitionSpec as PSpec

class DPModel(nn.Module):
    params: dict
    def get_input(self, coord, static_args, nbrs_nm):
        # =================================================================
        # MODIFICATION: Type-Agnostic Hack (Separate Geometry from Chemistry)
        # =================================================================
        # 1. Extract the STATIC real type indices using standard numpy
        real_type_idx_static = np.asarray(static_args['type_idx'])
        
        # 2. Create the fake indices as a STATIC numpy array
        # This prevents the TracerArrayConversionError in reorder_by_device!
        fake_type_idx = np.zeros_like(real_type_idx_static)
        
        # 3. Convert the real ones to JAX for the neural network embeddings later
        real_type_idx = jnp.asarray(real_type_idx_static)
        
        # 4. Force the model to see only 1 universal group of atoms
        type_count = np.array([coord.shape[0]])
        valid_types = np.array([0])
        nsel = [0] 
        
        compress = self.params.get('is_compressed', False)
        K = jax.device_count() if nbrs_nm is not None else 1
        
        # Use the STATIC fake types for physical reordering (NumPy logic)
        coord = reorder_by_device(coord, fake_type_idx, K=K)  
        
        if nbrs_nm is not None:
            type_count_new = [-(-type_count[i]//K) for i in range(len(type_count))]
            mask = get_mask_by_device(type_count)
            return coord, type_count_new, mask, compress, K, nsel, nbrs_nm, fake_type_idx, real_type_idx
        else:
            return coord, type_count, jnp.ones_like(coord[:,0]), compress, 1, nsel, None, fake_type_idx, real_type_idx
        #END OF MODIFICATION
            
    @nn.compact
    @nn.compact
    def __call__(self, coord_N3, box_33, static_args, nbrs_nm=None):
        # =================================================================
        # 1. SETUP: Separate Geometry (Fake) from Chemistry (Real)
        # =================================================================
        coord_N3, type_count, mask, compress, K, nsel, nbrs_nm, fake_type_idx, real_type_idx = self.get_input(coord_N3, static_args, nbrs_nm)
        A, L = self.params['axis'], static_args['lattice']['lattice_max'] if nbrs_nm is None else None
        
        ntypes_real = self.params['ntypes']
        type_embedding_weights = self.param('type_embedding', 
                                            lambda key: jnp.ones((ntypes_real, self.params['embed_type_width'])) * 0.1)

        # =================================================================
        # 2. GEOMETRY: Universal Distance & Tensor Calculation
        # NOTE: We now unpack idx_nm (3 items) from the modified get_relative_coord
        # =================================================================
        x_n3m, r_nm, idx_nm = get_relative_coord(coord_N3, box_33, type_count, static_args.get('lattice',None), nbrs_nm)
        sr_nm = [[sr(r, self.params['rcut']) for r in R] for R in r_nm]
        
        global_mean = jnp.mean(jnp.array(self.params['sr_mean']))
        global_std  = jnp.mean(jnp.array(self.params['sr_std']))
        
        sr_norm_nm = [[r/global_std for r in R] for R in sr_nm]
        sr_centernorm_nm = [[(r-global_mean)/global_std for r in R] for R in sr_nm]
        
        x_norm_n3m = [[x/(r+1e-16)[:,None] for x,r in zip(X,R)] for X,R in zip(x_n3m,r_nm)]
        R_n3m = [[3**0.5 * sr[:,None] * x for sr,x in zip(SR,X)] for SR,X in zip(sr_norm_nm,x_norm_n3m)]
        R_n4m = [[concat([sr[:,None],r], axis=1) for sr,r in zip(SR,R)] for SR,R in zip(sr_norm_nm,R_n3m)]
        
        R_nsel6m = [[3*sr[:,None]*tensor_3to6(x,axis=1,bias=1/3) for sr,x in zip(sr_norm_nm[0], x_norm_n3m[0])]]
        R_nselXm = [[concat([sr[:,None],r3] + ([r6] if self.params['use_2nd'] else []), axis=1)
                    for sr,r3,r6 in zip(sr_norm_nm[0], R_n3m[0], R_nsel6m[0])]]

        # =================================================================
        # 3. CHEMISTRY: Injecting the Embeddings (Type-Agnostic)
        # We extract the single universal tensors and inject chemical identity
        # =================================================================
        sr_vals = sr_centernorm_nm[0][0]  
        rx = R_nselXm[0][0]               
        neighbor_idx = idx_nm[0][0]       
        
        central_embed = type_embedding_weights[real_type_idx]
        neighbor_embed = type_embedding_weights[real_type_idx[neighbor_idx]]
        central_embed_tiled = jnp.tile(central_embed[:, None, :], (1, sr_vals.shape[1], 1))
        
        input_tensor = concat([sr_vals[:, :, None], central_embed_tiled, neighbor_embed], axis=2)

        # =================================================================
        # 4. NEURAL NETWORK: Embedding Net & Message Passing (Zero Loops!)
        # =================================================================
        if not self.params.get('use_mp', False):
            shared_embed = embedding_net(self.params['embed_widths'])
            embedded = shared_embed(input_tensor, compress, rx)
            T_NselXW = concat([embedded], K=K) / self.params['Nnbrs']
        else:
            C, E = self.params['embed_widths'][-1], self.params['embedMP_widths'][0]
            shared_embed_mp = embedding_net(self.params['embed_widths'] + (E,), out_linear_only=True)
            shared_embed_t2 = embedding_net(self.params['embed_widths'])
            
            F_ij = shared_embed_mp(input_tensor, compress) 
            
            r4 = R_n4m[0][0]
            T_2_n4C_outer = [] 
            for _ in range(2):
                embedded_t2 = shared_embed_t2(input_tensor, compress, r4)
                T_2_n4C_outer.append(embedded_t2 / self.params['Nnbrs'])
            
            T_2_nD = [(t[:,:,None]*t[:,:,:4,None]).sum(1).reshape(-1, 4*C) for t in T_2_n4C_outer]
            T_2_n3C = [t[:,1:] for t in T_2_n4C_outer]
            
            if nbrs_nm is not None:
                T_2_nD, T_2_n3C = lax.with_sharding_constraint([T_2_nD, T_2_n3C], PSpec())
                
            shared_lin_D_0 = linear_norm(E)
            shared_lin_D_1 = linear_norm(E)
            shared_lin_3C_0 = linear_norm(E)
            shared_lin_3C_1 = linear_norm(E)
            shared_layer_norm = self.param('shared_layer_norm', ones_init, (1,))**2 if self.params['atomic'] else 1
            
            # The complex Message Passing loops collapse into these vector operations:
            D_1_neighbor = shared_lin_D_1(T_2_nD[1])[neighbor_idx]
            C3_1_neighbor = shared_lin_3C_1(T_2_n3C[1])[neighbor_idx].transpose(0, 2, 1, 3)
            r3 = R_n3m[0][0]
            
            term_D = shared_lin_D_0(T_2_nD[0])[:, None, :] + D_1_neighbor
            term_3C = (r3[..., None] * (shared_lin_3C_0(T_2_n3C[0])[:, :, None, :] + C3_1_neighbor)).sum(1)
            F_msg = (term_D + term_3C + F_ij) * shared_layer_norm
            
            shared_embed_mp_final = embedding_net(self.params['embedMP_widths'], in_bias_only=True,
                                                dt_layers=range(2,len(self.params['embedMP_widths'])))
            T_NselXW_raw = shared_embed_mp_final(F_msg, reducer=rx)
            T_NselXW = concat([T_NselXW_raw], K=K) / self.params['Nnbrs']

        # =================================================================
        # 5. FITTING NET: Final Output Processing
        # =================================================================
        T_NselW = T_NselXW[:,0] + self.param('Tbias',zeros_init,T_NselXW.shape[-1:])
        T_Nsel3W = T_NselXW[:,1:4]
        T_Nsel6W = T_NselXW[:,4:] 
        
        G_NselAW = T_NselW[:,None]*T_NselW[:,:A,None] + (T_Nsel3W[:,:,None]*T_Nsel3W[:,:,:A,None]).sum(1)
        if self.params['use_2nd']:
            G2_axis_Nsel6A = tensor_3to6(T_Nsel3W[:,:,A:2*A], axis=1) + T_Nsel6W[:,:,A:2*A]
            G_NselAW += (G2_axis_Nsel6A[...,None] * T_Nsel6W[:,:,None]).sum(1)
            
        if not self.params['atomic']: 
            shared_fitting = fitting_net(self.params['fit_widths'])
            first_pred = shared_fitting(G_NselAW.reshape(G_NselAW.shape[0], -1))[:, 0]
            # Use real types to fetch the correct base energy!
            ebias_per_atom = jnp.array(self.params['Ebias'])[real_type_idx]
            pred = (mask * (first_pred + ebias_per_atom)).sum()
        else: 
            shared_fitting = fitting_net(self.params['fit_widths'], use_final=False)
            fit_all = shared_fitting(G_NselAW.reshape(G_NselAW.shape[0], -1))
            T_NselYW = (T_Nsel6W + tensor_3to6(T_Nsel3W, axis=1) + T_NselW[:,None] * jnp.array([1,1,1,0,0,0])[:,None]) if self.params['type'] == 'atomic_t2' else T_Nsel3W
            pred_raw = (fit_all[:, None] * T_NselYW).sum(-1)
            pred = concat([lax.with_sharding_constraint(pred_raw, PSpec()) if K > 1 else pred_raw], K=K)
            pred = pred[atomic_inverse_perm(fake_type_idx, [0])]
            if self.params['type'] == 'atomic_t2':
                s = 2**-0.5
                pred = pred[:, [0,4,3,4,1,5,3,5,2]] * jnp.array([1,s,s,s,1,s,s,s,1])
                
        debug = T_NselYW if self.params['atomic'] else T_NselXW
        return pred * self.params['out_norm'], debug

    def energy(self, variables, coord_N3, box_33, static_args, nbrs_nm=None):
        pred, _ = self.apply(variables, coord_N3, box_33, static_args, nbrs_nm)
        return pred

    def energy_and_force(self, variables, coord_N3, box_33, static_args, nbrs_nm=None):
        (pred, _), g = value_and_grad(self.apply, argnums=1, has_aux=True)(variables, coord_N3, box_33, static_args, nbrs_nm)
        return pred, -g
    
    def wc_predict(self, variables, coord_N3, box_33, static_args, nbrs_nm=None):
        wc_relative = self.apply(variables, coord_N3, box_33, static_args, nbrs_nm)[0]
        nsel_mask = np.isin(np.asarray(static_args['type_idx']), self.params['nsel'])
        return coord_N3[nsel_mask] + wc_relative
    
    def get_loss_fn(self, order='l2'):
        if self.params['atomic'] is False:
            vmap_energy_and_force = vmap(self.energy_and_force, (None, 0, 0, None))
            def loss_ef(variables, batch_data, pref, static_args):
                e, f = vmap_energy_and_force(variables, batch_data['coord'], batch_data['box'], static_args)
                if order == 'l2':
                    le = ((batch_data['energy'] - e)**2).mean() / (f.shape[1])**2
                    lf = ((batch_data['force'] - f)**2).mean()
                    return pref['e']*le + pref['f']*lf, (le, lf)
                elif order == 'l1-mixed':
                    le = jnp.abs(batch_data['energy'] - e).mean() / f.shape[1]
                    sq = ((batch_data['force'] - f)**2).mean(-1)
                    lf = jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, 1.)), 0.).mean()
                    return (pref['e']**0.5)*le + (pref['f']**0.5)*lf, (le, lf)
            loss_and_grad = value_and_grad(loss_ef, has_aux=True)
            return loss_ef, loss_and_grad
        else:
            vmap_apply = vmap(self.apply, (None, 0, 0, None))
            def loss_atomic(variables, batch_data, static_args):
                pred, _ = vmap_apply(variables, batch_data['coord'], batch_data['box'], static_args)
                if order == 'l2':
                    return ((batch_data['atomic'] - pred)**2).mean()
                elif order == 'l1-mixed':
                    sq = ((batch_data['atomic'] - pred)**2).mean(-1)
                    return jnp.where(sq > 0, jnp.sqrt(jnp.where(sq > 0, sq, 1.)), 0.).mean()
            loss_and_grad = value_and_grad(loss_atomic)
            return loss_atomic, loss_and_grad

    def get_observable_loss_fn(self):
        vmap_energy = vmap(self.energy, (None, 0, 0, None))
        def loss_obs(variables, batch_data, pref, static_args, temperature, target_observable):
            e = vmap_energy(variables, batch_data['coord'], batch_data['box'], static_args)
            kb = 8.617333262e-5
            beta = 1 / (kb * temperature)
            logweights = - beta * (e - batch_data['energy'])
            logweights -= jnp.amax(logweights)  # for numerical stability, we displace the exponents of the weights
            weights = jnp.exp(logweights)
            observable = batch_data['observable']
            if len(observable.shape) == 1:
                observable = observable[:, None]  # ensure observable is 2D
            obs_avg = jnp.sum(observable * weights[:, None], axis=0) / jnp.sum(weights) # observable reweighted expectation value
            lobs = jnp.mean((obs_avg - target_observable)**2)
            return pref['obs']*lobs, (lobs, obs_avg, observable, logweights)
        loss_and_grad = value_and_grad(loss_obs, has_aux=True)
        return loss_obs, loss_and_grad

