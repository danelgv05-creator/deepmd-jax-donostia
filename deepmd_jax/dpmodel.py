import jax.numpy as jnp
import jax
from jax import vmap, value_and_grad, lax
import flax.linen as nn
from .utils import *
from jax.sharding import PartitionSpec as PSpec

class DPModel(nn.Module):
    params: dict
    def get_input(self, coord, static_args, nbrs_nm):
        # MODIFICATION: Extract type_idx early for returning atomic type
        type_idx = np.asarray(static_args['type_idx'])
        # END MODIFICATION
        valid_types = self.params['valid_types']
        assert_type_idx_in_valid_types(type_idx, valid_types)
        #MODIFICATION: Fixed type_count indexing for mixed types - list([valid_types]) was wrong, should be list(valid_types)
        type_count = np.bincount(type_idx, minlength=self.params['ntypes'])[list(valid_types)]
        #END MODIFICATION
        compress = self.params.get('is_compressed', False)
        if self.params['atomic']:
            nsel = [list(valid_types).index(i) for i in self.params['nsel']]
        else:
            nsel = list(range(len(type_count)))
        K = jax.device_count() if nbrs_nm is not None else 1
        coord = reorder_by_device(coord, type_idx, K=K)  #Innecesary?
        if nbrs_nm is not None:
            nbrs_nm = [[nbrs_nm[i][j] for j in valid_types] for i in valid_types]
            type_count_new = [-(-type_count[i]//K) for i in range(len(type_count))]
            mask = get_mask_by_device(type_count)
            # MODIFICATION: Added type_idx to return statement
            return coord, type_count_new, mask, compress, K, nsel, nbrs_nm, type_idx
            # END MODIFICATION
        else:
            # MODIFICATION: Added type_idx to return statement
            return coord, type_count, jnp.ones_like(coord[:,0]), compress, 1, nsel, None, type_idx
            # END MODIFICATION
            
    @nn.compact
    def __call__(self, coord_N3, box_33, static_args, nbrs_nm=None):
        # prepare input parameters
        # MODIFICATION: Updated unpacking to include type_idx (atomic type)
        coord_N3, type_count, mask, compress, K, nsel, nbrs_nm, type_idx = self.get_input(coord_N3, static_args, nbrs_nm)
        # END MODIFICATION
        A, L = self.params['axis'], static_args['lattice']['lattice_max'] if nbrs_nm is None else None
        valid_types = np.asarray(self.params['valid_types'])
        #MODIFICATION: Atomic type always has to be determined.
        selected_types = valid_types[nsel] if self.params['atomic'] else valid_types
        
        # MODIFICATION: Build the chemical-type embedding once and reuse it for every species.
        ntypes = self.params['ntypes']
        type_embedding_weights = self.param('type_embedding',                        # Añdd type embedding to the learnable parameters
                                            lambda key: jnp.ones((ntypes,self.params['embed_type_width'])) * 0.1) #Initialize the net weights for each type
        #jax.debug.print("Type embedding weights: {weights}", weights=type_embedding_weights)
        # END MODIFICATION
        
        # compute relative coordinates x_3NM, distance r_NM, s(r) and normalized s(r)
        x_n3m, r_nm = get_relative_coord(coord_N3, box_33, type_count, static_args.get('lattice',None), nbrs_nm)
        sr_nm = [[sr(r, self.params['rcut']) for r in R] for R in r_nm]
        sr_norm_nm = [[r/std for r in R] for R,std in zip(sr_nm,self.params['sr_std'])]
        sr_centernorm_nm = [[(r-mean)/std for r in R] for R,mean,std in zip(sr_nm,self.params['sr_mean'],self.params['sr_std'])]
        # environment matrix: sr_norm_nm (0th-order), R_n3m (1st-order), R2_n6m (2nd-order)
        x_norm_n3m = [[x/(r+1e-16)[:,None] for x,r in zip(X,R)] for X,R in zip(x_n3m,r_nm)]
        R_n3m = [[3**0.5 * sr[:,None] * x for sr,x in zip(SR,X)] for SR,X in zip(sr_norm_nm,x_norm_n3m)]
        R_n4m = [[concat([sr[:,None],r], axis=1) for sr,r in zip(SR,R)] for SR,R in zip(sr_norm_nm,R_n3m)]
        R_nsel6m = [[3*sr[:,None]*tensor_3to6(x,axis=1,bias=1/3) for sr,x
                     in zip(sr_norm_nm[nsel[i]],x_norm_n3m[nsel[i]])] for i in range(len(nsel))]
        R_nselXm = [[concat([sr[:,None],r3] + ([r6] if self.params['use_2nd'] else []), axis=1)
                    for sr,r3,r6 in zip(sr_norm_nm[nsel[i]],R_n3m[nsel[i]],R_nsel6m[i])] for i in range(len(nsel))]
        # compute embedding net and atomic features T
        if not self.params.get('use_mp', False):                                          # original DP without message passing
            # MODIFICATION: Reuse one shared embedding network for all chemical species.
            shared_embed = embedding_net(self.params['embed_widths'])                     #General network G
            # MODIFICATION: Handle mixed types by processing each neighbor type separately
            type_idx_for_embed = selected_types if self.params['atomic'] else valid_types # Determine the type indices to use for embedding
            embedding_list = []                                                           # List to store embeddings for each type
            for i in range(len(nsel)):
                # MODIFICATION: For each selected type, sum contributions from all neighbor types
                type_embedding_parts = [] # List to store the embeddings of the different type from neighbours
                for j,(sr_vals,rx) in enumerate(zip(sr_centernorm_nm[nsel[i]], R_nselXm[i])):
                    # MODIFICATION: Expand type embeddings to match geometric tensor shapes (atoms, neighbors, features)
                    n_atoms, n_neighbors = sr_vals.shape[0], sr_vals.shape[1]                                                            # Number of central and neighbour atoms of the selected type
                    central_type_embed = jnp.tile(type_embedding_weights[type_idx_for_embed[i]][None,None,:], (n_atoms, n_neighbors, 1)) #Embedding for the central atom type
                    neighbor_type_embed = jnp.tile(type_embedding_weights[valid_types[j]][None,None,:], (n_atoms, n_neighbors, 1))       #Embedding for the neighbour atom type
                    input_tensor = concat([sr_vals[:,:,None], central_type_embed, neighbor_type_embed], axis=2)                          # Combine geometric and chemical information into the input tensor
                    embedded = shared_embed(input_tensor, compress, rx)                                                                  # Compute embedding for this neighbour type
                    type_embedding_parts.append(embedded)                                                                                # Accumulate embeddings from all neighbor types for this central atom type
                # END MODIFICATION
                type_embedding = sum(type_embedding_parts)                # Summ contributions of different neighbour typer for the current central atom type
                embedding_list.append(type_embedding)                     # Accumulate the embedding for this selected type
            T_NselXW = concat(embedding_list, K=K) / self.params['Nnbrs'] # Final embedding for all selected types, averaged by number of neighbors
            # END MODIFICATION

        else: # Message Passing: Compute atomic features T; linear transform, add into F; Y=#types; B=2C, D=4C
            C, E = self.params['embed_widths'][-1], self.params['embedMP_widths'][0]
            #MODIFICATION: Share the embedding stages across species, with chemistry injected through type embeddings.
            shared_embed_mp = embedding_net(self.params['embed_widths'] + (E,), out_linear_only=True) #One layer embedding for the escalar message 
            shared_embed_t2 = embedding_net(self.params['embed_widths'])                              #General network G
            #MODIFICATION: Process each selected type separately for mixed-type support
            embed_nselmE_list = []
            for i in range(len(nsel)):
                #MODIFICATION: For each selected type, process neighbor types and expand type embeddings
                embed_type_parts = []
                for j, sr_vals in enumerate(sr_centernorm_nm[nsel[i]]):
                    #MODIFICATION: Expand type embeddings to match geometric tensor shapes (atoms, neighbors, features)
                    n_atoms, n_neighbors = sr_vals.shape[0], sr_vals.shape[1]                                                        
                    central_type_embed = jnp.tile(type_embedding_weights[selected_types[i]][None,None,:], (n_atoms, n_neighbors, 1)) 
                    neighbor_type_embed = jnp.tile(type_embedding_weights[valid_types[j]][None,None,:], (n_atoms, n_neighbors, 1))   
                    input_tensor = concat([sr_vals[:,:,None], central_type_embed, neighbor_type_embed], axis=2)                      
                    embedded = shared_embed_mp(input_tensor, compress)                                                               
                    embed_type_parts.append(embedded)                                                                                
                #END MODIFICATION
                embed_nselmE_list.append(embed_type_parts) # List of embeddings for each selected type, where each element contains the contributions from all neighbor types
            embed_nselmE = embed_nselmE_list 
            
            #MODIFICATION: Compute the atomic features T4 with the general embedding, enriched with chemical information in the input.
            T_2_n4C_outer = [] 
            for _ in range(2):
                T_2_n4C_inner = []
                for outer_i, (SR, R4_vals) in enumerate(zip(sr_centernorm_nm, R_n4m)):
                    #MODIFICATION: For each type, sum contributions from all neighbor types with proper shape handling
                    type_contributions = []
                    for j, (sr_vals, r4) in enumerate(zip(SR, R4_vals)):
                        #MODIFICATION: Expand type embeddings to match shapes
                        n_atoms, n_neighbors = sr_vals.shape[0], sr_vals.shape[1]
                        central_type_embed = jnp.tile(type_embedding_weights[valid_types[outer_i]][None,None,:], (n_atoms, n_neighbors, 1))
                        neighbor_type_embed = jnp.tile(type_embedding_weights[valid_types[j]][None,None,:], (n_atoms, n_neighbors, 1)) #Cuantas veces multiplica cada eje
                        input_tensor = concat([sr_vals[:,:,None], central_type_embed, neighbor_type_embed], axis=2)
                        embedded = shared_embed_t2(input_tensor, compress, r4)
                        type_contributions.append(embedded)
                    #END MODIFICATION
                    T_2_n4C_inner.append(sum(type_contributions) / self.params['Nnbrs'])
                T_2_n4C_outer.append(T_2_n4C_inner)
            T_2_n4C = T_2_n4C_outer
            #END MODIFICATION
            T_2_nD = [[(t[:,:,None]*t[:,:, :4,None]).sum(1).reshape(-1,4*C) for t in T] for T in T_2_n4C]
            T_2_n3C = [[t[:,1:] for t in T] for T in T_2_n4C]
            if nbrs_nm is not None:
                T_2_nD, T_2_n3C = lax.with_sharding_constraint([T_2_nD, T_2_n3C], PSpec())
            #Compute the message F for each pair of types, using the shared embedding's output.
            F_nselmE = [[(linear_norm(E)(T_2_nD[0][i])[:,None]
                      + (linear_norm(E)(T_2_nD[1][j])[nbrs_nm[i][j]] if nbrs_nm is not None else
                         jnp.repeat(linear_norm(E)(T_2_nD[1][j]),L,axis=0))
                      + (R_n3m[i][j][...,None] * (linear_norm(E)(T_2_n3C[0][i])[:,:,None]
                          + (linear_norm(E)(T_2_n3C[1][j])[nbrs_nm[i][j]].transpose(0,2,1,3) if nbrs_nm is not None else
                            jnp.repeat(linear_norm(E)(T_2_n3C[1][j]),L,axis=0).transpose(1,0,2)))).sum(1)
                      + emb) * (self.param('layer_norm_%d_%d'%(i,j), ones_init, (1,))**2 if self.params['atomic'] else 1)
                        for j,emb in enumerate(EMB)] for i,EMB in zip(nsel,embed_nselmE)]
            #MODIFICATION: Compute the final embedding once and reuse for all types
            shared_embed_mp_final = embedding_net(self.params['embedMP_widths'], in_bias_only=True,
                                                dt_layers=range(2,len(self.params['embedMP_widths'])))

            #MODIFICATION: Reuse the final embedding for all selected types
            T_NselXW_list = []
            for F, RX in zip(F_nselmE, R_nselXm):
                type_final_embedding = sum([shared_embed_mp_final(f, reducer=rx) for f, rx in zip(F, RX)])
                T_NselXW_list.append(type_final_embedding)
            T_NselXW = concat(T_NselXW_list, K=K) / self.params['Nnbrs']
            #END MODIFICATION
            #jpesos_mp = shared_embed_mp_final.variables.get('params', {})
            
            #jax.debug.print("MP embedding weights: {mpweights}", mpweights=jpesos_mp)

        # compute fitting net with input G = T @ T_sub; energy is sum of output; A for any axis dimension
        T_NselW, T_Nsel3W, T_Nsel6W = T_NselXW[:,0]+self.param('Tbias',zeros_init,T_NselXW.shape[-1:]), T_NselXW[:,1:4], T_NselXW[:,4:] 
        G_NselAW = T_NselW[:,None]*T_NselW[:,:A,None] + (T_Nsel3W[:,:,None]*T_Nsel3W[:,:,:A,None]).sum(1)
        if self.params['use_2nd']:
            G2_axis_Nsel6A = tensor_3to6(T_Nsel3W[:,:,A:2*A], axis=1) + T_Nsel6W[:,:,A:2*A]
            G_NselAW += (G2_axis_Nsel6A[...,None] * T_Nsel6W[:,:,None]).sum(1)
        if not self.params['atomic']: # Energy prediction
            # MODIFICACIÓN: use a sungle shared fitting net for all types
            shared_fitting = fitting_net(self.params['fit_widths'])# Initialize a single general net
            first_pred = shared_fitting(G_NselAW.reshape(G_NselAW.shape[0], -1))[:, 0] # Pass every atom through the shared fitting net
            ebias_por_atomo = jnp.repeat(jnp.array(self.params['Ebias']), type_count)  # Create an array of Ebias values for each atom based on its type
            pred = (mask * (first_pred + ebias_por_atomo)).sum()                       #Sum the energies  and appy the mask
            
        else: # Atomic tensor prediction
            # MODIFICACIÓN: use a single shared fitting net
            sel_count = [type_count[i] for i in nsel] #Count each atom type number
            shared_fitting = fitting_net(self.params['fit_widths'], use_final=False)# Initialize a single general net            
            fit_all = shared_fitting(G_NselAW.reshape(G_NselAW.shape[0], -1))       # Pass G through the fitting net
            fit_nselW = split(fit_all, sel_count, 0, K=K)                           #Divide the result into atomic types, to mantain type_pred compatibility
            if self.params['type'] == 'atomic_t2':
                T_NselYW = (T_Nsel6W + tensor_3to6(T_Nsel3W, axis=1) + T_NselW[:,None] * jnp.array([1,1,1,0,0,0])[:,None])
            elif self.params['type'] == 'atomic':
                T_NselYW = T_Nsel3W
            else:  # MODIFICATION: Fallback for unexpected types
                T_NselYW = T_Nsel3W
            T_nselYW = split(T_NselYW, sel_count, 0, K=K)
            #MODIFICATION: Fixed indexing for mixed types - use actual selected type indices
            real_type_count = np.bincount(static_args['type_idx'], minlength=self.params['ntypes'])
            pred_list = []
            for i, (f, T, nsel_i) in enumerate(zip(fit_nselW, T_nselYW, nsel)):
                # Compute prediction for atoms of this selected type
                type_pred = (f[:,None]*T).sum(-1)[:real_type_count[nsel_i]]
                pred_list.append(type_pred)
            #END MODIFICATION
            pred = concat([lax.with_sharding_constraint(p, PSpec()) if K > 1 else p for p in pred_list])
            #MODIFICATION: Updated reordering to handle mixed types using actual selected type indices
            pred = pred[atomic_inverse_perm(static_args['type_idx'], nsel)]
            #END MODIFICATION
            if self.params['type'] == 'atomic_t2': # tensor_6to9
                s = 2**-0.5
                pred = pred[:, [0,4,3,4,1,5,3,5,2]] * jnp.array([1,s,s,s,1,s,s,s,1])
            debug = T_NselYW
        if not self.params['atomic']:
            debug = T_NselXW
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

