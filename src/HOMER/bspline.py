import numpy as np
import jax 
import jax.numpy as jnp

class jax_spline_not_a_knot:
    def __init__(self, m, n):
        # passing matrix
        phi=np.zeros(((n+2)*(m+2),(m+2)*(n+2)))
        # interpolation equations 
        for j in range(m):
            for i in range(n):
                phi[i+(j)*n,i+(j)*(n+2)]=1
                phi[i+(j)*n,i+(j)*(n+2)+1]=4
                phi[i+(j)*n,i+(j)*(n+2)+2]=1
                phi[i+(j)*n,i+(j+1)*(n+2)]=4
                phi[i+(j)*n,i+(j+1)*(n+2)+1]=16
                phi[i+(j)*n,i+(j+1)*(n+2)+2]=4
                phi[i+(j)*n,i+(j+2)*(n+2)]=1
                phi[i+(j)*n,i+(j+2)*(n+2)+1]=4
                phi[i+(j)*n,i+(j+2)*(n+2)+2]=1
        # y- border
        for i in range(m):
            phi[n*m+i,(i+1)*(n+2)]=1
            phi[n*m+i,(i+1)*(n+2)+1]=-2
            phi[n*m+i,(i+1)*(n+2)+2]=1
        # y+ border
        for i in range(m):
            phi[n*m+m+i,(i+1)*(n+2)+(n+2-3)]=1
            phi[n*m+m+i,(i+1)*(n+2)+(n+2-2)]=-2
            phi[n*m+m+i,(i+1)*(n+2)+(n+2-1)]=1
        # x+ border
        for i in range(n):
            phi[n*m+2*m+i,1+i]=1
            phi[n*m+2*m+i,1+i+n+2]=-2
            phi[n*m+2*m+i,1+i+2*(n+2)]=1
        # x- border
        for i in range(n):
            phi[n*m+2*m+n+i,-1-(1+i)]=1
            phi[n*m+2*m+n+i,-1-(1+i+(n+2))]=-2
            phi[n*m+2*m+n+i,-1-(1+i+2*(n+2))]=1      
        # x- y- corner   
        phi[n*m+2*m+2*n,0]=1
        phi[n*m+2*m+2*n,n+2+1]=-2
        phi[n*m+2*m+2*n,2*(n+2)+2]=1
        # x- y+ corner   
        phi[n*m+2*m+2*n+1,(n+2)-1]=1
        phi[n*m+2*m+2*n+1,(n+2)+((n+2)-1)-1]=-2
        phi[n*m+2*m+2*n+1,2*(n+2)+((n+2)-2)-1]=1
        # x+ y- corner    
        phi[n*m+2*m+2*n+2,(m+2)*(n+2)-(n+2)]=1
        phi[n*m+2*m+2*n+2,(m+2)*(n+2)-2*(n+2)+1]=-2
        phi[n*m+2*m+2*n+2,(m+2)*(n+2)-3*(n+2)+2]=1
        # x+ y+ corner    
        phi[n*m+2*m+2*n+3,(m+2)*(n+2)-2*(n+2)-3]=1
        phi[n*m+2*m+2*n+3,(m+2)*(n+2)-(n+2)-2]=-2
        phi[n*m+2*m+2*n+3,(m+2)*(n+2)-1]=1
        # control points 
        phi_inv=np.linalg.inv(phi)
        self.phi = phi
        self.phi_inv = phi_inv
        self.m = m
        self.n = n


    def eval(self, knot_params, eval_locs):
        x,y,z = knot_params.T
        Px=jnp.concatenate((x,jnp.tile(jnp.zeros(1),(self.m+2)*(self.n+2)-(self.m*self.n))))
        Py=jnp.concatenate((y,jnp.tile(jnp.zeros(1),(self.m+2)*(self.n+2)-(self.m*self.n))))
        Pz=jnp.concatenate((z,jnp.tile(jnp.zeros(1),(self.m+2)*(self.n+2)-(self.m*self.n))))

        Qx=36*self.phi_inv @ Px
        Qy=36*self.phi_inv @ Py
        Qz=36*self.phi_inv @ Pz

        u, v = eval_locs.T
        mu = u % 1
        mv = v % 1

        V1=(1-mv)**3
        V2=3*mv**3-6*mv**2+4
        V3=-3*mv**3+3*mv**2+3*mv+1
        V4=mv**3
        U1=(1-mu)**3    
        U2=3*mu**3-6*mu**2+4
        U3=-3*mu**3+3*mu**2+3*mu+1
        U4=mu**3

        pu = (u - mu).astype(int)
        pv = (v - mv).astype(int)
        n = self.n

        param_x=((V1*(Qx[pv+0+pu*(n+2)]*U1+Qx[pv+0+n+2+pu*(n+2)]*U2+Qx[pv+0+2*(n+2)+pu*(n+2)]*U3+Qx[pv+0+3*(n+2)+pu*(n+2)]*U4)) 
        	+(V2*(Qx[pv+1+pu*(n+2)]*U1+Qx[pv+1+n+2+pu*(n+2)]*U2+Qx[pv+1+2*(n+2)+pu*(n+2)]*U3+Qx[pv+1+3*(n+2)+pu*(n+2)]*U4)) 
        	+(V3*(Qx[pv+2+pu*(n+2)]*U1+Qx[pv+2+n+2+pu*(n+2)]*U2+Qx[pv+2+2*(n+2)+pu*(n+2)]*U3+Qx[pv+2+3*(n+2)+pu*(n+2)]*U4)) 
        	+(V4*(Qx[pv+3+pu*(n+2)]*U1+Qx[pv+3+n+2+pu*(n+2)]*U2+Qx[pv+3+2*(n+2)+pu*(n+2)]*U3+Qx[pv+3+3*(n+2)+pu*(n+2)]*U4)))/36

        param_y=((V1*(Qy[pv+0+pu*(n+2)]*U1+Qy[pv+0+n+2+pu*(n+2)]*U2+Qy[pv+0+2*(n+2)+pu*(n+2)]*U3+Qy[pv+0+3*(n+2)+pu*(n+2)]*U4)) 
        	+(V2*(Qy[pv+1+pu*(n+2)]*U1+Qy[pv+1+n+2+pu*(n+2)]*U2+Qy[pv+1+2*(n+2)+pu*(n+2)]*U3+Qy[pv+1+3*(n+2)+pu*(n+2)]*U4)) 
        	+(V3*(Qy[pv+2+pu*(n+2)]*U1+Qy[pv+2+n+2+pu*(n+2)]*U2+Qy[pv+2+2*(n+2)+pu*(n+2)]*U3+Qy[pv+2+3*(n+2)+pu*(n+2)]*U4)) 
        	+(V4*(Qy[pv+3+pu*(n+2)]*U1+Qy[pv+3+n+2+pu*(n+2)]*U2+Qy[pv+3+2*(n+2)+pu*(n+2)]*U3+Qy[pv+3+3*(n+2)+pu*(n+2)]*U4)))/36  

        param_z=((V1*(Qz[pv+0+pu*(n+2)]*U1+Qz[pv+0+n+2+pu*(n+2)]*U2+Qz[pv+0+2*(n+2)+pu*(n+2)]*U3+Qz[pv+0+3*(n+2)+pu*(n+2)]*U4)) 
       		+(V2*(Qz[pv+1+pu*(n+2)]*U1+Qz[pv+1+n+2+pu*(n+2)]*U2+Qz[pv+1+2*(n+2)+pu*(n+2)]*U3+Qz[pv+1+3*(n+2)+pu*(n+2)]*U4)) 
       		+(V3*(Qz[pv+2+pu*(n+2)]*U1+Qz[pv+2+n+2+pu*(n+2)]*U2+Qz[pv+2+2*(n+2)+pu*(n+2)]*U3+Qz[pv+2+3*(n+2)+pu*(n+2)]*U4)) 
       		+(V4*(Qz[pv+3+pu*(n+2)]*U1+Qz[pv+3+n+2+pu*(n+2)]*U2+Qz[pv+3+2*(n+2)+pu*(n+2)]*U3+Qz[pv+3+3*(n+2)+pu*(n+2)]*U4)))/36

        return jnp.column_stack((param_x, param_y, param_z))

    def deriv_u(self, params, eval_locs):
        def loc_u(eval_a, eval_b):
            return self.eval(params, jnp.column_stack((eval_a, eval_b))).squeeze()
        jax_fn = jax.vmap(jax.jacfwd(loc_u, argnums=0))
        return jax_fn(eval_locs[:, 0], eval_locs[:, 1])

    def deriv_v(self, params, eval_locs):
        def loc_u(eval_a, eval_b):
            return self.eval(params, jnp.column_stack((eval_a, eval_b))).squeeze()
        jax_fn = jax.vmap(jax.jacfwd(loc_u, argnums=1))
        return jax_fn(eval_locs[:, 0], eval_locs[:, 1])

    def deriv_mat(self, params, eval_locs):
        return jnp.concatenate((
            self.deriv_u(params, eval_locs)[..., None],
            self.deriv_v(params, eval_locs)[..., None],
            ), axis=-1)

    def normal(self, params, eval_locs):
        n =  jnp.cross(
            self.deriv_u(params, eval_locs)[:, None],
            self.deriv_v(params, eval_locs)[:, None]
        )
        return n / jnp.linalg.norm(n, axis=-1, keepdims=True)

    def local_u_coords(self, params, eval_locs):
        u = self.deriv_u(params, eval_locs)
        u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
        n = self.normal(params, eval_locs).squeeze()
        m = jnp.cross(u, n)
        m = m / jnp.linalg.norm(m, axis=-1, keepdims=True)
        coord_mat = jnp.concatenate((u[..., None], m[..., None], n[..., None]), axis=-1)
        return jnp.linalg.inv(coord_mat)
        breakpoint()

    def local_u_deriv_mat(self, params, eval_locs):
        coord_shift = self.local_u_coords(params, eval_locs)
        d_mat = self.deriv_mat(params, eval_locs)
        new_coords= coord_shift @ d_mat
        return new_coords[:, :2, :]





