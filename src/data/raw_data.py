from dataclasses import dataclass
import numpy as np
from numpy import ndarray

import os
from typing import Union, List, Tuple

from .exporter import Exporter

from ..tokenizer.spec import DetokenizeOutput
from .order import Order

@dataclass(frozen=True)
class RawData(Exporter):
    '''
    Dataclass to handle data from processed model files.
    '''
    
    # vertices of the mesh, shape (N, 3)
    vertices: Union[ndarray, None]
    
    # normals of vertices, shape (N, 3)
    vertex_normals: Union[ndarray, None]
    
    # faces of mesh, shape (F, 3), face id starts from 0 to F-1
    faces: Union[ndarray, None]
    
    # face normal of mesh, shape (F, 3)
    face_normals: Union[ndarray, None]
    
    # joints of bones, shape (J, 3)
    joints: Union[ndarray, None]
    
    # skinning of joints, shape (N, J)
    skin: Union[ndarray, None]
    
    # parents of joints, None represents no parent(a root joint)
    # make sure parent[k] < k
    parents: Union[List[Union[int, None]], None]
    
    # names of joints
    names: Union[List[str], None]
    
    # local coordinate
    matrix_local: Union[ndarray, None]
    
    # tails of joints, shape (J, 3)
    tails: Union[ndarray, None]=None
    
    # whether the joint has skin, bool
    no_skin: Union[ndarray, None]=None
    
    # path to data
    path: Union[str, None]=None
    
    # data cls
    cls: Union[str, None]=None
    
    @staticmethod
    def load(path: str, origin=np.float16, to=np.float32) -> 'RawData':
        data = np.load(path, allow_pickle=True)
        d = {name: data[name][()] for name in data}
        d['path'] = path
        skin = d.get('skin', None)
        if skin is not None:
            d['no_skin'] = ~np.any(skin>0, axis=0)
        else:
            d['no_skin'] = None
        return RawData(**d).change_dtype(origin, to)
    
    def change_dtype(self, origin, to) -> 'RawData':
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ndarray) and v.dtype == origin:
                v = v.astype(to)
            d[k] = v
        return RawData(**d)

    def add_breast_bones(self) -> 'RawData':
        if self.names is None or self.joints is None:
            return self
            
        if 'J_Sec_L_Bust1' in self.names or 'J_Sec_R_Bust1' in self.names:
            return self
            
        chest_names = ['J_Bip_C_UpperChest', 'J_Bip_C_Chest']
        chest_idx = -1
        for name in chest_names:
            if name in self.names:
                chest_idx = self.names.index(name)
                break
                
        if chest_idx == -1:
            return self
            
        # Try to find shoulders to get width
        l_shoulder_idx = self.names.index('J_Bip_L_Shoulder') if 'J_Bip_L_Shoulder' in self.names else -1
        r_shoulder_idx = self.names.index('J_Bip_R_Shoulder') if 'J_Bip_R_Shoulder' in self.names else -1
        neck_idx = self.names.index('J_Bip_C_Neck') if 'J_Bip_C_Neck' in self.names else -1
        
        chest_pos = self.joints[chest_idx]
        
        # We need Z to be forward, Y to be up, X to be right.
        if l_shoulder_idx != -1 and r_shoulder_idx != -1:
            l_pos = self.joints[l_shoulder_idx]
            r_pos = self.joints[r_shoulder_idx]
            
            # Distance between shoulders
            width = np.linalg.norm(l_pos - r_pos)
            
            # Direction from Right to Left shoulder
            x_dir = (l_pos - r_pos)
            x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-6)
            
            # Up direction (from chest to neck if possible)
            if neck_idx != -1:
                neck_pos = self.joints[neck_idx]
                y_dir = neck_pos - chest_pos
                spine_len = np.linalg.norm(y_dir)
                if spine_len > 1e-6:
                    y_dir = y_dir / spine_len
                else:
                    y_dir = np.array([0., 1., 0.])
            else:
                y_dir = np.array([0., 1., 0.])
            
            # Forward direction
            z_dir = np.cross(x_dir, y_dir)
            z_dir = z_dir / (np.linalg.norm(z_dir) + 1e-6)
        else:
            width = 0.3
            x_dir = np.array([1., 0., 0.])
            y_dir = np.array([0., 1., 0.])
            z_dir = np.array([0., 0., 1.])
            
        # Positions for breasts (approximate)
        # Left bust: move slightly left, down, and forward from upper chest
        # Increased the offset multipliers based on typical anatomy
        l_bust_pos = chest_pos + x_dir * (width * 0.22) - y_dir * (width * 0.15) + z_dir * (width * 0.25)
        # Right bust: move slightly right, down, and forward from upper chest
        r_bust_pos = chest_pos - x_dir * (width * 0.22) - y_dir * (width * 0.15) + z_dir * (width * 0.25)
        
        # Tails (extend slightly forward and down)
        l_bust_tail = l_bust_pos - y_dir * (width * 0.05) + z_dir * (width * 0.15)
        r_bust_tail = r_bust_pos - y_dir * (width * 0.05) + z_dir * (width * 0.15)
        
        # Append new bones
        new_names = list(self.names)
        new_names.extend(['J_Sec_L_Bust1', 'J_Sec_R_Bust1'])
        
        new_parents = list(self.parents)
        new_parents.extend([chest_idx, chest_idx])
        
        new_joints = np.concatenate([self.joints, np.array([l_bust_pos, r_bust_pos], dtype=self.joints.dtype)], axis=0)
        
        if self.tails is not None:
            new_tails = np.concatenate([self.tails, np.array([l_bust_tail, r_bust_tail], dtype=self.tails.dtype)], axis=0)
        else:
            new_tails = None
            
        # Update kwargs
        kwargs = dict(self.__dict__)
        kwargs['names'] = new_names
        kwargs['parents'] = new_parents
        kwargs['joints'] = new_joints
        kwargs['tails'] = new_tails
        
        if self.no_skin is not None:
            new_no_skin = np.concatenate([self.no_skin, np.array([False, False], dtype=self.no_skin.dtype)], axis=0)
            kwargs['no_skin'] = new_no_skin
            
        if self.skin is not None:
            new_skin = np.concatenate([self.skin, np.zeros((self.skin.shape[0], 2), dtype=self.skin.dtype)], axis=1)
            
            # --- SKINNING LOGIC ---
            # We want to transfer some weight from the chest to the new bust bones for nearby vertices.
            if self.vertices is not None:
                # Calculate distance from vertices to breast bones
                l_dist = np.linalg.norm(self.vertices - l_bust_pos, axis=1)
                r_dist = np.linalg.norm(self.vertices - r_bust_pos, axis=1)
                
                # Determine an influence radius based on shoulder width
                radius = width * 0.35
                
                # Weight transfer function (linear decay based on distance)
                # Max transfer weight at center = 0.8 (leaves 20% to chest)
                l_weight_transfer = np.clip(1.0 - (l_dist / radius), 0, 1) * 0.8
                r_weight_transfer = np.clip(1.0 - (r_dist / radius), 0, 1) * 0.8
                
                # Ensure left/right don't overlap too much by taking the max if they do,
                # but typically they won't overlap heavily.
                
                # Transfer from chest
                chest_weights = self.skin[:, chest_idx]
                
                l_transferred = chest_weights * l_weight_transfer
                r_transferred = chest_weights * r_weight_transfer
                
                new_skin[:, chest_idx] = np.clip(chest_weights - l_transferred - r_transferred, 0, 1)
                new_skin[:, -2] = l_transferred
                new_skin[:, -1] = r_transferred
                
            kwargs['skin'] = new_skin
            
        if self.matrix_local is not None:
            l_matrix = np.eye(4, dtype=self.matrix_local.dtype)
            r_matrix = np.eye(4, dtype=self.matrix_local.dtype)
            
            l_matrix[:3, 3] = l_bust_pos - chest_pos
            r_matrix[:3, 3] = r_bust_pos - chest_pos
            
            new_matrix = np.concatenate([self.matrix_local, np.array([l_matrix, r_matrix], dtype=self.matrix_local.dtype)], axis=0)
            kwargs['matrix_local'] = new_matrix

        return RawData(**kwargs)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(file=path, **self.__dict__)
    
    @property
    def N(self):
        '''
        number of vertices
        '''
        return self.vertices.shape[0]
    
    @property
    def F(self):
        '''
        number of faces
        '''
        return self.faces.shape[0]
    
    @property
    def J(self):
        '''
        number of joints
        '''
        return self.joints.shape[0]
    
    def check(self):
        if self.names is not None and self.joints is not None:
            assert len(self.names) == self.J
        if self.names is not None and self.parents is not None:
            assert len(self.names) == len(self.parents)
        if self.parents is not None:
            for (i, pid) in enumerate(self.parents):
                if i==0:
                    assert pid is None
                else:
                    assert pid is not None
                    assert pid < i
    
    def export_pc(self, path: str, with_normal: bool=True, normal_size=0.01):
        '''
        export point cloud
        '''
        if with_normal:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=self.vertex_normals, normal_size=normal_size)
        else:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=None, normal_size=normal_size)
    
    def export_mesh(self, path: str):
        '''
        export mesh
        '''
        self._export_mesh(vertices=self.vertices, faces=self.faces, path=path)
    
    def export_skeleton(self, path: str):
        '''
        export spring
        '''
        self._export_skeleton(joints=self.joints, parents=self.parents, path=path)
    
    def export_skeleton_sequence(self, path: str):
        '''
        export spring
        '''
        self._export_skeleton_sequence(joints=self.joints, parents=self.parents, path=path)
    
    def export_fbx(
        self,
        path: str,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        use_tail: bool=False,
        custom_vertex_group: Union[ndarray, None]=None,
    ):
        '''
        export the whole model with skining
        '''
        self._export_fbx(
            path=path,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin if custom_vertex_group is None else custom_vertex_group,
            parents=self.parents,
            names=self.names,
            faces=self.faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
            tails=self.tails if use_tail else None,
        )
    
    def export_render(self, path: str, resolution: Tuple[int, int]=[256, 256]):
        self._export_render(
            path=path,
            vertices=self.vertices,
            faces=self.faces,
            bones=np.concatenate([self.joints, self.tails], axis=-1),
            resolution=resolution,
        )

@dataclass(frozen=True)
class RawSkeleton(Exporter):
    '''
    Dataclass to handle skeleton from AR.
    '''
    # joints of bones, shape (J, 3), float32
    joints: Union[ndarray, None]
    
    # tails of joints, shape (J, 3), float32
    tails: Union[ndarray, None]
    
    # whether the joint has skin, bool
    no_skin: Union[ndarray, None]
    
    # parents of joints, None represents no parent(a root joint)
    # make sure parent[k] < k
    parents: Union[List[Union[int, None]], None]
    
    # names of joints
    names: Union[List[str], None]
    
    @staticmethod
    def load(path: str) -> 'RawSkeleton':
        data = np.load(path, allow_pickle=True)
        return RawSkeleton(**{name: data[name][()] for name in data})
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(file=path, **self.__dict__)
    
    @staticmethod
    def from_detokenize_output(res: DetokenizeOutput, order: Union[Order, None]) -> 'RawSkeleton':
        J = len(res.bones)
        names = order.make_names(cls=res.cls, parts=res.parts, num_bones=J)
        joints = res.joints
        p_joints = res.p_joints
        parents = []
        for (i, joint) in enumerate(joints):
            if i == 0:
                parents.append(None)
                continue
            p_joint = p_joints[i]
            dis = 999999
            pid = None
            for j in reversed(range(i)):
                n_dis = ((joints[j] - p_joint)**2).sum()
                if n_dis < dis:
                    pid = j
                    dis = n_dis
            parents.append(pid)
        return RawSkeleton(
            joints=joints,
            tails=res.tails,
            no_skin=res.no_skin,
            parents=parents,
            names=names,
        )
        
    def export_skeleton(self, path: str):
        '''
        export spring
        '''
        self._export_skeleton(joints=self.joints, parents=self.parents, path=path)
    
    def export_skeleton_sequence(self, path: str):
        '''
        export spring
        '''
        self._export_skeleton_sequence(joints=self.joints, parents=self.parents, path=path)
    
    def export_fbx(
        self,
        path: str,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        use_tail: bool=False,
    ):
        '''
        export the whole model with skining
        '''
        self._export_fbx(
            path=path,
            vertices=None,
            joints=self.joints,
            skin=None,
            parents=self.parents,
            names=self.names,
            faces=None,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            use_extrude_bone=use_extrude_bone,
            use_connect_unique_child=use_connect_unique_child,
            extrude_from_parent=extrude_from_parent,
            tails=self.tails if use_tail else None,
        )
    
    def export_render(self, path: str, resolution: Tuple[int, int]=[256, 256]):
        self._export_render(
            path=path,
            vertices=None,
            faces=None,
            bones=np.concatenate([self.joints, self.tails], axis=-1),
            resolution=resolution,
        )

@dataclass
class RawSkin(Exporter):
    '''
    Dataclass to handle skeleton from AR.
    '''
    # skin, shape (J, N)
    skin: ndarray
    
    # always sampled, shape (N, 3)
    vertices: Union[ndarray, None]=None
    
    # for future use, shape (J, 3)
    joints: Union[ndarray, None]=None
    
    @staticmethod
    def load(path: str) -> 'RawSkin':
        data = np.load(path, allow_pickle=True)
        return RawSkin(**{name: data[name][()] for name in data})
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(file=path, **self.__dict__)