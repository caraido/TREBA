import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
import tqdm
from numba import jit

FRAME_WIDTH_TOP=1280
FRAME_HEIGHT_TOP=1024
# SVD paths
svd_computer_path = 'util/datasets/Schwartz_mouse_v2/svd/Schwartz_mouse_svd_computer.pickle'
mean_path = 'util/datasets/Schwartz_mouse_v2/svd/Schwartz_mouse_mean.pickle'

def normalize(data:np.ndarray,bodyparts:list):
    """Scale by the median length/original length."""
    shape=data.shape
    flat_data=data.reshape(-1, shape[-1]) # last dimension is bodyparts
    nose_x_i=bodyparts.index('nose_x')
    nose_y_i=bodyparts.index('nose_y')
    tailbase_x_i=bodyparts.index('tail_base_x')
    tailbase_y_i=bodyparts.index('tail_base_y')

    body_length=np.linalg.norm(np.array([flat_data[:,nose_x_i]-flat_data[:,tailbase_x_i],
                               flat_data[:,nose_y_i]-flat_data[:,tailbase_y_i]]), axis=0)
    body_length_median=np.median(body_length)
    flat_data=flat_data/body_length_median
    new_data=flat_data.reshape(shape)
    return new_data

def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[2] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.multiply(data, scale) + shift


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


# to calculate angle between 2 vectors for 2D or 3D
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    assert v1.shape==v2.shape
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if len(v1.shape)==1:
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    elif v1.shape[0]==2 or v1.shape[0]==3:
        return np.arccos(np.clip(np.einsum('ij,ij->j',v1_u, v2_u), -1.0, 1.0))
    else:
        return np.arccos(np.clip(np.einsum('ij,ij->j', v1_u.T, v2_u.T), -1.0, 1.0))



# Helper function to return indexes of nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


# Interpolates all nan values of given array
def interpol(trajectories):
    for i,tra in enumerate(trajectories):
        for j,arr in enumerate(np.transpose(tra)):
           #TODO: interpolate the data

            trajectories[i,:,j]=arr
    return trajectories

# for each seq
@jit(nopython=True)
def single_alignment_2D(trajectory: np.ndarray, t_matrix, r_matrix):
    # here we need to make (x,y,1) from (x,y)
    # the output will be (x*,y*), which will be a vector of 16 instead of 24.
    seq_shape = trajectory.shape
    new_trajectory = np.zeros((seq_shape[0], int(2 * seq_shape[1] / 3)))
    # TODO: double check this one
    for i in range(int(seq_shape[1] / 3)):
        joint=np.ones((3,seq_shape[0]))
        joint[0] = trajectory[:,3 * i]
        joint[1] = trajectory[:,3*i+1]
        aligned = np.dot(r_matrix, np.dot(t_matrix, joint))
        normalized_aligned = aligned / aligned[-1]
        new_trajectory[:,2 * i:2 * i + 2] = normalized_aligned[:-1,:].T

    return new_trajectory

# for each frame
# TODO: need to change it to each seq
def single_alignment_3D(trajectory: np.ndarray, t_matrix, r_matrix):
    # here we need to make (x,y,z,1) from (x,y,z)
    # the output will be (x*,y*,z*), which will be a vector of 24 instead of 48
    # TODO: not sure what are the {joint}_v_{x,y,z} is
    new_trajectory = np.zeros([1, int(len(trajectory) / 2)])
    for i in range(int(len(trajectory) / 6)):
        joint = np.array([trajectory[6 * i], trajectory[6 * i + 1], trajectory[6 * i + 2], 1])
        aligned = np.dot(r_matrix, np.dot(t_matrix, joint))
        normalized_aligned = aligned / aligned[-1]
        new_trajectory[3 * i:3 * i + 2] = normalized_aligned[:-1]

    return new_trajectory


# for egocentric alignment, we need
# the undistorted trajectories time series for all the joints 2D or 3D
# body parts
# which split the output:
# rotational matrix
# translational matrix
# ego-centric trajectories
def alignment(trajectories: np.ndarray, bodyparts: list):
    '''
    here the trajectories are seq_num x seq_length(default 21) x ego_centric body part
    '''
    # for 2D there is
    if len(bodyparts) == 24:
        assert 'skull_base_x' in bodyparts
        assert 'skull_base_y' in bodyparts

        skull_base_x_i=bodyparts.index('skull_base_x')
        skull_base_y_i=bodyparts.index('skull_base_y')

        nose_x_i = bodyparts.index('nose_x')
        nose_y_i = bodyparts.index('nose_y')
        tail_base_x_i = bodyparts.index('tail_base_x')
        tail_base_y_i = bodyparts.index('tail_base_y')

        # the neck base will be pinned to the origin of the coordinate
        all_aligned = []
        all_t_matrix = []
        all_r_matrix = []

        for tra in trajectories:
            # this t_matrix defines the translation from allocentric to egocentric based on the first frame in a seq
            # making it a square matrix so that it is inversible for future use
            # translational matrix for the first frame
            t_matrix = np.array([[1, 0, -tra[0][skull_base_x_i]],
                                 [0, 1, -tra[0][skull_base_y_i]],
                                 [0, 0, 1]])

            # the nose-to-tail-base line will be aligned to y axis and nose will point to the positive direction
            tail2nose_x = tra[:,nose_x_i] - tra[:,tail_base_x_i]
            tail2nose_y = tra[:,nose_y_i] - tra[:,tail_base_y_i]
            tail2nose = np.array([tail2nose_x, tail2nose_y])
            # align to y axis.
            angle = np.pi/2-np.arctan2(tail2nose[1,0], tail2nose[0,0])

            # making it a square matrix so that it is inversible for future use
            # rotational matrix for the first frame
            r_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0],
                                 [0, 0, 1]])
            aligned = single_alignment_2D(tra, t_matrix, r_matrix)

            all_aligned.append(aligned)
            all_t_matrix.append(t_matrix)
            all_r_matrix.append(r_matrix)
            new_bodyparts = [x for x in bodyparts if "likelihood" not in x]

        return all_aligned, all_t_matrix, all_r_matrix, new_bodyparts
    # for 3D there is
    # TODO: need to rewrite. Refer to 2D alignemnt code
    elif len(bodyparts) == 48:
        assert 'left_ear_x' in bodyparts
        assert 'left_ear_y' in bodyparts
        assert 'left_ear_z' in bodyparts

        left_ear_x_i = bodyparts.index('left_ear_x')
        left_ear_y_i = bodyparts.index('left_ear_y')
        left_ear_z_i = bodyparts.index('left_ear_z')
        right_ear_x_i = bodyparts.index('right_ear_x')
        right_ear_y_i = bodyparts.index('right_ear_y')
        right_ear_z_i = bodyparts.index('right_ear_z')

        nose_x_i = bodyparts.index('nose_x')
        nose_y_i = bodyparts.index('nose_y')
        nose_z_i = bodyparts.index('nose_y')
        tail_base_x_i = bodyparts.index('tail_base_x')
        tail_base_y_i = bodyparts.index('tail_base_y')
        tail_base_z_i = bodyparts.index('tail_base_z')

        all_aligned = []
        all_t_matrix = []
        all_r_matrix = []
        for tra in trajectories:
            neck_center_x = (tra[left_ear_x_i] + tra[right_ear_x_i]) / 2
            neck_center_y = (tra[left_ear_y_i] + tra[right_ear_y_i]) / 2
            neck_center_z = (tra[left_ear_z_i] + tra[right_ear_z_i]) / 2
            # this t_matrix defines the translation from allocentric to egocentric
            # making it a square matrix so that it is inversible for future use
            t_matrix = np.array([[1, 0, 0, -neck_center_x],
                                 [0, 1, 0, -neck_center_y],
                                 [0, 0, 1, -neck_center_z],
                                 [0, 0, 0, 1]])
            # the nose-to-tail-base line will be aligned to y axis and nose will point to the positive direction
            tail2nose_x = tra[nose_x_i] - tra[tail_base_x_i]
            tail2nose_y = tra[nose_y_i] - tra[tail_base_y_i]
            tail2nose_z = tra[nose_z_i] - tra[tail_base_z_i]

            tail2nose = np.array([tail2nose_x, tail2nose_y, tail2nose_z])
            tail2nose_2D = np.array([tail2nose_x, tail2nose_y])

            z_pos = np.array([0, 0, 1])
            y_pos = np.array([0, 1])

            angle = angle_between(tail2nose, z_pos)
            z_angle = np.pi / 2 - angle
            y_angle = np.arctan2(tail2nose_2D, y_pos)

            # making it a square matrix so that it is inversible for future use
            # two rotations are involved
            # first one rotation to x-y plane
            # second one rotation to align to y axis
            r_matrix_z = np.array([[np.cos(z_angle), -np.sin(z_angle), 0, 0],
                                   [np.sin(z_angle), np.cos(z_angle), 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])
            r_matrix_to_y = np.array([[np.cos(y_angle), -np.sin(y_angle), 0, 0],
                                      [np.sin(y_angle), np.cos(y_angle), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
            r_matrix = np.dot(r_matrix_to_y, r_matrix_z)

            # (x*,y*,z*,1)=R*T*(x,y,z,1)
            # the 1 in the output is emitted
            aligned = single_alignment_3D(tra, t_matrix, r_matrix)
            all_aligned.append(aligned)
            all_t_matrix.append(t_matrix)
            all_r_matrix.append(r_matrix)

        all_aligned = np.array(all_aligned)
        all_t_matrix = np.array(all_t_matrix)
        all_r_matrix = np.array(all_r_matrix)
        new_bodyparts = [x for x in bodyparts if "v" not in x] # we don't need the velocity information
        return all_aligned, all_t_matrix, all_r_matrix, new_bodyparts
    else:
        raise Exception('check the shape of the trajectories! It should be either T*24 or T*48')

def alignment_wrapper(data:dict,body_parts:list,remove:list):
    new_data={}
    all_t_matrix={}
    all_r_matrix={}
    print("\naligning the data:")
    for key,value in tqdm.tqdm(data.items()):
        aligned,t_matrix,r_matrix,new_bodyparts=alignment(value,body_parts)
        aligned=np.array(aligned)
        # check if the bodypart
        new_aligned=[]
        bodyparts_reborn=[]
        for i,bp in enumerate(new_bodyparts):
            contain=sum([bp.__contains__(r) for r in remove])
            if not contain:
                new_aligned.append(aligned[:,:,i])
                bodyparts_reborn.append(bp)
        new_aligned=np.array(new_aligned).reshape(aligned.shape[0],aligned.shape[1],-1)
        new_data[key]=np.array(new_aligned)
        all_t_matrix[key]=t_matrix
        all_r_matrix[key]=r_matrix

    return new_data,all_t_matrix,all_r_matrix,bodyparts_reborn

def mirror_wrapper(data:dict,body_parts:list):
    new_data = {}
    print('\n mirroring the data')
    for key, value in data.items():
        mirrorred = mirror(value, body_parts)
        new_data[key] = mirrorred
    return new_data

# mirror all the data to one direction
def mirror(trajectories:np.ndarray, bodyparts):
    '''
    here the trajectories are egocentric seq_num x seq_length(default 21) x body part
    bodypart should be new body parts without likelilood or velocity
    '''
    # for 2D
    nose_x_i = bodyparts.index('nose_x')
    tail_base_x_i = bodyparts.index('tail_base_x')

    new_trajectories=[]

    for tra in tqdm.tqdm(trajectories):

        tail2nose_x=tra[:,nose_x_i]-tra[:,tail_base_x_i]

        difference=tail2nose_x[-1]-tail2nose_x[0]
        if difference<0:
            for i, bodypart in enumerate(bodyparts):
                if '_x' in bodypart:
                    tra[:,i]=-tra[:,i]
        new_trajectories.append(tra)

    return np.array(new_trajectories)

def unnormalize_keypoint_center_rotation(keypoints, center, rotation):

    keypoints = keypoints.reshape((-1, int(keypoints.shape[-1]/2), 2))

    # Apply inverse rotation
    rotation = -1 * rotation
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation),  np.cos(rotation)]]).transpose((2, 0, 1))
    centered_data = np.matmul(R, keypoints.transpose(0, 2, 1))

    keypoints = centered_data + center[:, :, np.newaxis]
    keypoints = keypoints.transpose(0, 2, 1)

    return keypoints.reshape(-1, int(keypoints.shape[-2]*2))


def transform_to_svd_components(data,bodyparts,
                                center_index=3,
                                n_components=5,
                                svd_computer=None,
                                mean=None,
                                stack_agents = False,
                                stack_axis = 1,
                                save_svd = False):
    # data shape is num_seq x 1 x 8 x 2
    #resident_keypoints = data[:, 0, :, :]
    #intruder_keypoints = data[:, 1, :, :]
    #data = np.concatenate([resident_keypoints, intruder_keypoints], axis=0)
    data=np.squeeze(data)
    # Center the data using given center_index
    mouse_center = data[:, center_index, :]
    centered_data = data - mouse_center[:, np.newaxis, :]

    # Rotate such that keypoints 3 and 4 are parallel with the y axis
    mouse_rotation = np.arctan2(
        data[:, int(bodyparts.index('nose_x')/2), 0] - data[:, int(bodyparts.index('tail_base_x')/2), 0],
        data[:, int(bodyparts.index('nose_x')/2), 1] - data[:, int(bodyparts.index('tail_base_x')/2), 1])

    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                   [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1)))

    # Encode mouse rotation as sine and cosine
    mouse_rotation = np.concatenate([np.sin(mouse_rotation)[:, np.newaxis], np.cos(
        mouse_rotation)[:, np.newaxis]], axis=-1)

    centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
    centered_data = centered_data.transpose((0, 2, 1))

    centered_data = centered_data.reshape((-1, len(bodyparts)))

    if mean is None:
        mean = np.mean(centered_data, axis=0)
    centered_data = centered_data - mean

    # Compute SVD components
    if svd_computer is None:
        svd_computer = TruncatedSVD(n_components=n_components)
        svd_data = svd_computer.fit_transform(centered_data)
    else:
        svd_data = svd_computer.transform(centered_data)
        explained_variances = np.var(svd_data, axis=0) / np.var(centered_data, axis=0).sum()

    # Concatenate state as mouse center, mouse rotation and svd components
    data = np.concatenate([mouse_center, mouse_rotation, svd_data], axis=1)

    if save_svd:
        with open(svd_computer_path, 'wb') as f:
            pickle.dump(svd_computer, f)
        with open(mean_path, 'wb') as f:
            pickle.dump(mean, f)

    return data, svd_computer, mean


def transform_svd_to_keypoints(data, svd_computer, mean, stack_agents = False,
                            stack_axis = 0):
    # here the input size: (frame_num * (4+n_components))
    num_components = data.shape[1]
    center = data[:, :2]
    rotation = data[:, 2:4]
    components = data[:, 4:num_components]

    keypoints = svd_computer.inverse_transform(components)

    if mean is not None:
        keypoints = keypoints + mean

    # Compute rotation angle from sine and cosine representation
    rotation = np.arctan2(
        rotation[:, 0], rotation[:, 1])

    keypoints = unnormalize_keypoint_center_rotation(
        keypoints, center, rotation)

    return keypoints