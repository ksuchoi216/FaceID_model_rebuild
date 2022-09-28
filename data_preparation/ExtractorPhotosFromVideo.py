import os
import cv2


class ExtractorPhotosFromVideo(object):
    def __init__(self, cfg):
        IsSavePhoto = cfg['IsSavePhoto']
        IsSaveFace = cfg['IsSaveFace']
        
        folder_name_for_source = cfg['folder_name_for_source']
        folder_name_for_video = cfg['folder_name_for_video']
        folder_name_to_save_photos = cfg['folder_name_to_save_photos']
        user_names = cfg['user_names']
        video_names = cfg['video_names']
        video_extension = cfg['video_extension']
        
        frame_count_limit = cfg['frame_count_limit']
        frame_interval_for_sampling = cfg['frame_interval_for_sampling']

        source_path = './'+folder_name_for_source
        path_for_photo = os.path.join(source_path,
                                      folder_name_to_save_photos)
        if IsSavePhoto and not os.path.exists(path_for_photo):
            os.makedirs(path_for_photo)
        print(path_for_photo)

        for user_name, video_name in zip(user_names, video_names):
            path_for_video = os.path.join(
                source_path,
                folder_name_for_video,
                video_name + video_extension
            )
            path_for_username_folder = os.path.join(path_for_photo,
                                                    user_name)
            print(f'{path_for_video} with {path_for_username_folder}')
        
            if not os.path.exists(path_for_username_folder):
                os.makedirs(path_for_username_folder)
            
            vc = cv2.VideoCapture(path_for_video)
            success, frame = vc.read()
            
            if not success:
                print(f"Capturing frames failed from {path_for_video}")

            count = 0
            frame_count = 0

            while success:
                success, frame = vc.read()
                if frame is not None:
                    h, w, ch = frame.shape

                frame = frame[int(h*(2/8)):int(h*(8/8)),
                              int(w*(2/8)): int(w*(6/8))]

                if frame_count == frame_count_limit:
                    break

                count += 1
                if count % frame_interval_for_sampling == 0:
                    path_for_save = os.path.join(
                        path_for_username_folder,
                        str(frame_count) + '.jpg'
                    )
                    cv2.imwrite(path_for_save, frame)
                    success, frame = vc.read()
                    msg = (
                        f'[{count:5}][{frame_count:4}] '
                        f'A image was saved at {path_for_save}'
                    )
                    print(msg)
                    frame_count += 1