import argparse
import pathlib

import cv2 as cv
import h5py
import numpy as np
import pandas as pd
import scipy.io
import tqdm


def convert_pose(vector: np.ndarray) -> np.ndarray:
    rot = cv.Rodrigues(np.array(vector).astype(np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw]).astype(np.float32)


def convert_gaze(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw]).astype(np.float32)


def get_eval_info(person_id: str, eval_dir: pathlib.Path) -> pd.DataFrame:
    eval_path = eval_dir / f'{person_id}.txt'
    df = pd.read_csv(eval_path,
                     delimiter=' ',
                     header=None,
                     names=['path', 'side'])
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df


def save_one_person(person_id: str, data_dir: pathlib.Path,
                    eval_dir: pathlib.Path, output_path: pathlib.Path,
                    is_eval: bool) -> None:
    left_images = dict()
    left_poses = dict()
    left_gazes = dict()
    right_images = dict()
    right_poses = dict()
    right_gazes = dict()
    filenames = dict()
    person_dir = data_dir / person_id
    for path in sorted(person_dir.glob('*')):
        mat_data = scipy.io.loadmat(path.as_posix(),
                                    struct_as_record=False,
                                    squeeze_me=True)
        data = mat_data['data']

        day = path.stem
        left_images[day] = data.left.image
        left_poses[day] = data.left.pose
        left_gazes[day] = data.left.gaze

        right_images[day] = data.right.image
        right_poses[day] = data.right.pose
        right_gazes[day] = data.right.gaze

        filenames[day] = mat_data['filenames']

        if not isinstance(filenames[day], np.ndarray):
            left_images[day] = np.array([left_images[day]])
            left_poses[day] = np.array([left_poses[day]])
            left_gazes[day] = np.array([left_gazes[day]])
            right_images[day] = np.array([right_images[day]])
            right_poses[day] = np.array([right_poses[day]])
            right_gazes[day] = np.array([right_gazes[day]])
            filenames[day] = np.array([filenames[day]])

    df = get_eval_info(person_id, eval_dir)
    images = []
    poses = []
    gazes = []
    if is_eval:
        for _, row in df.iterrows():
            day = row.day
            index = np.where(filenames[day] == row.filename)[0][0]
            if row.side == 'left':
                image = left_images[day][index]
                pose = convert_pose(left_poses[day][index])
                gaze = convert_gaze(left_gazes[day][index])
            else:
                image = right_images[day][index][:, ::-1]
                pose = convert_pose(right_poses[day][index]) * np.array([1, -1])
                gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1])
            images.append(image)
            poses.append(pose)
            gazes.append(gaze)
    else:
        num_image = 300
        for _, row in df.iterrows():
            if np.random.rand() < 0.5:
                day = row.day
                index = np.where(filenames[day] == row.filename)[0][0]
                if row.side == 'left':
                    image = left_images[day][index]
                    pose = convert_pose(left_poses[day][index])
                    gaze = convert_gaze(left_gazes[day][index])
                else:
                    image = right_images[day][index][:, ::-1]
                    pose = convert_pose(right_poses[day][index]) * np.array([1, -1])
                    gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1])
                images.append(image)
                poses.append(pose)
                gazes.append(gaze)
                num_image -= 1
            if num_image <= 0:
                break
        # for path in sorted(person_dir.glob('*')):
        #     day = path.stem
        #     for index in range(len(filenames[day])):
        #         if np.random.rand() < 0.25:
        #             image = left_images[day][index]
        #             pose = convert_pose(left_poses[day][index])
        #             gaze = convert_gaze(left_gazes[day][index])
        #             images.append(image)
        #             poses.append(pose)
        #             gazes.append(gaze)
        #             num_image -= 1
        #         if np.random.rand() < 0.25:
        #             image = right_images[day][index][:, ::-1]
        #             pose = convert_pose(right_poses[day][index]) * np.array([1, -1])
        #             gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1])
        #             images.append(image)
        #             poses.append(pose)
        #             gazes.append(gaze)
        #             num_image -= 1
        #         if num_image <= 0:
        #             break
    images = np.asarray(images).astype(np.uint8)
    poses = np.asarray(poses).astype(np.float32)
    gazes = np.asarray(gazes).astype(np.float32)
    with h5py.File(output_path, 'a') as f_output:
        f_output.create_dataset(f'{person_id}/image', data=images)
        f_output.create_dataset(f'{person_id}/pose', data=poses)
        f_output.create_dataset(f'{person_id}/gaze', data=gazes)


def save_all(data_dir: pathlib.Path, eval_dir: pathlib.Path, output_path: pathlib.Path,
             is_eval: bool) -> None:
    images = []
    poses = []
    gazes = []
    for id in range(15):
        left_images = dict()
        left_poses = dict()
        left_gazes = dict()
        right_images = dict()
        right_poses = dict()
        right_gazes = dict()
        filenames = dict()

        person_id = f'p{id:02}'
        person_dir = data_dir / person_id
        for path in sorted(person_dir.glob('*')):
            mat_data = scipy.io.loadmat(path.as_posix(),
                                        struct_as_record=False,
                                        squeeze_me=True)
            data = mat_data['data']

            day = path.stem
            left_images[day] = data.left.image
            left_poses[day] = data.left.pose
            left_gazes[day] = data.left.gaze

            right_images[day] = data.right.image
            right_poses[day] = data.right.pose
            right_gazes[day] = data.right.gaze

            filenames[day] = mat_data['filenames']

            if not isinstance(filenames[day], np.ndarray):
                left_images[day] = np.array([left_images[day]])
                left_poses[day] = np.array([left_poses[day]])
                left_gazes[day] = np.array([left_gazes[day]])
                right_images[day] = np.array([right_images[day]])
                right_poses[day] = np.array([right_poses[day]])
                right_gazes[day] = np.array([right_gazes[day]])
                filenames[day] = np.array([filenames[day]])

        df = get_eval_info(person_id, eval_dir)
        if is_eval:
            for _, row in df.iterrows():
                day = row.day
                index = np.where(filenames[day] == row.filename)[0][0]
                if row.side == 'left':
                    image = left_images[day][index]
                    pose = convert_pose(left_poses[day][index])
                    gaze = convert_gaze(left_gazes[day][index])
                else:
                    image = right_images[day][index][:, ::-1]
                    pose = convert_pose(right_poses[day][index]) * np.array([1, -1])
                    gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1])
                images.append(image)
                poses.append(pose)
                gazes.append(gaze)
        else:
            for path in sorted(person_dir.glob('*')):
                day = path.stem
                for index in range(len(filenames[day])):
                    image = left_images[day][index]
                    pose = convert_pose(left_poses[day][index])
                    gaze = convert_gaze(left_gazes[day][index])
                    images.append(image)
                    poses.append(pose)
                    gazes.append(gaze)

                    image = right_images[day][index][:, ::-1]
                    pose = convert_pose(right_poses[day][index]) * np.array([1, -1])
                    gaze = convert_gaze(right_gazes[day][index]) * np.array([1, -1])
                    images.append(image)
                    poses.append(pose)
                    gazes.append(gaze)

    images = np.asarray(images).astype(np.uint8)
    poses = np.asarray(poses).astype(np.float32)
    gazes = np.asarray(gazes).astype(np.float32)
    with h5py.File(output_path, 'a') as f_output:
        f_output.create_dataset('image', data=images)
        f_output.create_dataset('pose', data=poses)
        f_output.create_dataset('gaze', data=gazes)

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--is-eval', type=int, required=True)
    parser.add_argument('--is-all', type=int, required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    is_eval = args.is_eval
    is_all = args.is_all
    if is_eval:
        if is_all:
            output_path = output_dir / 'MPIIGazeCalibrationAllEval.h5'
        else:
            output_path = output_dir / 'MPIIGazeCalibrationPerEval0.h5'
    else:
        if is_all:
            output_path = output_dir / 'MPIIGazeCalibrationAll.h5'
        else:
            output_path = output_dir / 'MPIIGazeCalibrationPer300.h5'
    if output_path.exists():
        raise ValueError(f'{output_path} already exists.')

    dataset_dir = pathlib.Path(args.dataset)

    data_dir = dataset_dir / 'Data' / 'Normalized'
    eval_dir = dataset_dir / 'Evaluation Subset' / 'sample list for eye image'
    if is_all:
        for i in tqdm.tqdm(range(1)):
            save_all(data_dir, eval_dir, output_path, is_eval=is_eval)
    else:
        for person_id in tqdm.tqdm(range(15)):
            person_id = f'p{person_id:02}'
            save_one_person(person_id, data_dir, eval_dir, output_path, is_eval=is_eval)


if __name__ == '__main__':
    main()
