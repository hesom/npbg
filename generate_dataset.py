import argparse
import os, sys
import yaml

import numpy as np
import glob
from tqdm import tqdm
from glumpy import app

from npbg.gl.render import OffscreenRender, create_shared_texture, cpy_tensor_to_buffer, cpy_tensor_to_texture
from npbg.gl.programs import NNScene
from npbg.gl.utils import load_scene_data, get_proj_matrix, crop_intrinsic_matrix, crop_proj_matrix, setup_scene, rescale_K, to_numpy, cv2_write
from npbg.gl.dataset import parse_input_string
import npbg.gl.nn as nn_
from npbg.datasets.common import load_image


def get_args():
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--config', type=str, default=None, required=True, help='config path')
    # parser.add_argument('--viewport', type=str, default='', help='width,height')
    # parser.add_argument('--mode', type=str, default='dataset', choices=['dataset', 'infer', 'overlay'])
    # parser.add_argument('--inputs', type=str, default='colors_p1', help='render what, like colors_ps10')
    # parser.add_argument('--use-mesh', action='store_true')
    # parser.add_argument('--use-texture', action='store_true')
    # parser.add_argument('--view-matrix', type=str, help='override view_matrix from config')

    # parser.add_argument('--render-desc', action='store_true', help='render point descriptors instead of rgb')
    
    # parser.add_argument('--pad_to', default='', type=str, help='width,height')
    # parser.add_argument('--out_extension', default='png')
    # parser.add_argument('--cam_as_name', action='store_true')
    # parser.add_argument('--save-dir', type=str, default='rendered')
    # parser.add_argument('--cameras_subset', type=str, default=None, help=' (optional) path to a text file with a subset '
    #     'of camera names to process. Each line in file may contain either camera name, or any filepath with a basename '
    #     'without extension representing camera name (the latter is done for conveniency).')
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--keepdir', action='store_true')
    # parser.add_argument('--append', action='store_true', help='append data to existing .npz files (forces --keepdir)')

    # parser.add_argument('--position-shift', type=str, default='0,0,0')

    # parser.add_argument('--cpu', action='store_true')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default=None, required=True, help='config path')
    parser.add_argument('--viewport', type=str, default='', help='width,height')
    parser.add_argument('--keep-fov', action='store_true', help='keep field of view when resizing viewport')
    parser.add_argument('--mode', type=str, default='dataset', choices=['dataset', 'infer', 'net_input', 'overlay'])
    parser.add_argument('--inputs', type=str, default='colors_p1', help='render what, like colors_ps10')
    parser.add_argument('--use-mesh', action='store_true')
    parser.add_argument('--use-texture', action='store_true')
    parser.add_argument('--view-matrix', type=str, help='override view_matrix from config')
    parser.add_argument('--pad_to', default='', type=str, help='width,height')
    parser.add_argument('--out_extension', default='png')
    parser.add_argument('--cam_as_name', action='store_true')
    parser.add_argument('--save-dir', type=str, default='rendered')
    parser.add_argument('--cameras_subset', type=str, default=None, help=' (optional) path to a text file with a subset '
        'of camera names to process. Each line in file may contain either camera name, or any filepath with a basename '
        'without extension representing camera name (the latter is done for conveniency).')
    # parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sweep-dir', action='store_true')
    parser.add_argument('--append', action='store_true', help='append data to existing .npz files')
    parser.add_argument('--position-shift', type=str, default='0,0,0')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--clear-color', type=str)
    parser.add_argument('--use-light', action='store_true')
    parser.add_argument('--temp-avg', action='store_true')

    args = parser.parse_args()

    args.viewport = [int(x) for x in args.viewport.split(',')] if args.viewport else None
    args.pad_to = [int(x) for x in args.pad_to.split(',')] if args.pad_to else None
    args.inputs = args.inputs.split(',')

    if args.append:
        args.keepdir = True

    return args


def write_images(name, dct, pad_to=None, out_extension='png', name_only=False):
    scene = args.config.split('/')[-1].split('.')[0]
    scene_name = f'{scene}_{name}'
    for key, value in dct.items():
        if len(value.shape) == 3 and value.shape[2] in (3, 4):
            if pad_to is not None:
                orig_h, orig_w = value.shape[:2]
                value = np.pad(value, ((0, pad_to[0] - orig_h), (0, pad_to[1] - orig_w), (0, 0)), mode='constant', 
                               constant_values=0)
            if name_only:
                cv2_write(f'{args.save_dir}/{name}.' + out_extension, value)
            else:
                cv2_write(f'{args.save_dir}/{scene_name}_{key}.' + out_extension, value)


def fix_viewport_size(viewport_size, factor=16):
    viewport_w = factor * (viewport_size[0] // factor)
    viewport_h = factor * (viewport_size[1] // factor)
    return viewport_w, viewport_h


def write_npz(name, dct, compressed=False):
    out = f'{args.save_dir}/{name}.npz'
    if compressed:
        np.savez_compressed(out, **dct)
    else:
        if args.append and os.path.exists(out):
            dct_l = np.load(out)
            dct.update(dct_l)
        np.savez(out, **dct)


def write_data(name, dct):
    # if args.debug:
    write_images(name, dct)
    # else:
    #     write_npz(name, dct)


if __name__ == '__main__':
    # window = get_window(visible=False) # need to create window for opengl

    # args = get_args()
    # assert not args.render_desc, 'FIX RENDER BUG'

    # with open(args.config) as f:
    #     _config = yaml.load(f)
    #     # support two types of configs
    #     # 1 type - config with scene data
    #     # 2 type - config with model checkpoints and path to scene data config
    #     if 'scene' in _config: # 1 type
    #         scene_data = load_scene_data(_config['scene'])
    #         net_ckpt = _config.get('net_ckpt')
    #         texture_ckpt = _config.get('texture_ckpt') 
    #     else:
    #         scene_data = load_scene_data(args.config)
    #         net_ckpt = scene_data['config'].get('net_ckpt')
    #         texture_ckpt = scene_data['config'].get('texture_ckpt')

    # args.use_mesh = args.use_mesh or _config.get('use_mesh') or args.use_texture

    # scene = NNScene()
    # setup_scene(scene, scene_data, args.use_mesh, args.use_texture)
    # # scene.set_model_view(np.eye(4))

    # out_buffer_location = 'numpy'

    # viewport_size = args.viewport if args.viewport else scene_data['config']['viewport_size']
    # print('viewport size ', viewport_size)
    # off_render = OffscreenRender(viewport_size=viewport_size, out_buffer_location=out_buffer_location)

    # if not args.keepdir:
    #     try:
    #         import shutil
    #         shutil.rmtree(args.save_dir)
    #     except:
    #         pass
    # os.makedirs(args.save_dir, exist_ok=True)

    # if scene_data['proj_matrix'] is not None:
    #     proj_matrix = scene_data['proj_matrix']
    # else:
    #     K_crop = crop_intrinsic_matrix(scene_data['intrinsic_matrix'], scene_data['config']['viewport_size'], viewport_size)
    #     proj_matrix = get_proj_matrix(K_crop, viewport_size)

    # if proj_matrix.shape[0] > 4:
    #     proj_matrix = proj_matrix.reshape(-1, 4, 4)

    # image_sizes = scene_data['image_sizes'] if 'image_sizes' in scene_data else None

    # ms_model=None
    # if args.mode == 'infer':
    #     model=nn_.OGL(scene, scene_data, viewport_size, net_ckpt, texture_ckpt, out_buffer_location=out_buffer_location,
    #             gpu=not args.cpu)

    args = get_args()

    with open(args.config) as f:
        _config = yaml.load(f)
        # support two types of configs
        # 1 type - config with scene data
        # 2 type - config with model checkpoints and path to scene data config
        if 'scene' in _config: # 1 type
            scene_data = load_scene_data(_config['scene'])
            net_ckpt = _config.get('net_ckpt')
            texture_ckpt = _config.get('texture_ckpt') 
        else:
            scene_data = load_scene_data(args.config)
            net_ckpt = scene_data['config'].get('net_ckpt')
            texture_ckpt = scene_data['config'].get('texture_ckpt')

    args.use_mesh = args.use_mesh or _config.get('use_mesh') or args.use_texture

    scene = NNScene()
    setup_scene(scene, scene_data, args.use_mesh, args.use_texture)
    # scene.set_model_view(np.eye(4))

    out_buffer_location = 'numpy'

    viewport_size = args.viewport if args.viewport else scene_data['config']['viewport_size']
    viewport_size = fix_viewport_size(viewport_size)
    print('viewport size ', viewport_size)

    window = app.Window(width=viewport_size[0], height=viewport_size[1], visible=False, fullscreen=False)
    off_render = OffscreenRender(viewport_size=viewport_size, out_buffer_location=out_buffer_location, clear_color=args.clear_color)

    if args.sweep_dir:
        try:
            import shutil
            shutil.rmtree(args.save_dir)
        except:
            pass
    os.makedirs(args.save_dir, exist_ok=True)

    if scene_data['intrinsic_matrix'] is not None:
        K_src = scene_data['intrinsic_matrix']
        old_size = scene_data['config']['viewport_size']
        sx = viewport_size[0] / old_size[0]
        sy = viewport_size[1] / old_size[1]
        K_crop = rescale_K(K_src, sx, sy, keep_fov=args.keep_fov)
        scene_data['proj_matrix'] = get_proj_matrix(K_crop, viewport_size)
    elif scene_data['proj_matrix'] is not None:
        new_proj_matrix = crop_proj_matrix(scene_data['proj_matrix'], *scene_data['config']['viewport_size'], *viewport_size)
        scene_data['proj_matrix'] = new_proj_matrix
    else:
        raise Exception('no intrinsics are provided')

    if scene_data['proj_matrix'].shape[0] > 4:
        scene_data['proj_matrix'] = scene_data['proj_matrix'].reshape(-1, 4, 4)

    image_sizes = scene_data['image_sizes'] if 'image_sizes' in scene_data else None

    ms_model=None
    if args.mode in ('infer', 'net_input'):
        model=nn_.OGL(scene, scene_data, viewport_size, net_ckpt, texture_ckpt, 
                      out_buffer_location=out_buffer_location,
                      gpu=not args.cpu, temporal_average=args.temp_avg)

    if args.view_matrix:
        view_matrix = np.loadtxt(args.view_matrix).reshape(-1, 4, 4)
        camera_labels = [str(i) for i in range(len(view_matrix))]
    else:
        view_matrix = scene_data['view_matrix']
        camera_labels = scene_data['camera_labels']
    # if not args.render_images:
    #     i_list = range(len(view_matrix))
    # else:
    #     rng = args.render_images.split(',')
    #     if len(rng) == 3:
    #         i_list = range(int(rng[0]), int(rng[1]), int(rng[2]))
    #     elif len(rng) == 2:
    #         i_list = range(int(rng[0]), int(rng[1]))
    #     else:
    #         i_list = [int(rng[0])]
    # print(i_list)
    if not args.cameras_subset:
        cameras_subset = set(camera_labels)
    else:
        lines = open(args.cameras_subset).readlines()
        lines = [os.path.splitext(os.path.split(record)[-1])[0] for record in lines]
        print('lines read from val file:')
        print(lines)

        cameras_subset = set(lines)
        print(f'Subset of {len(cameras_subset)} has been selected.')
    
    pos_shift = [float(x) for x in args.position_shift.split(',')]
    print('POSITION SHIFT', pos_shift)

    print('CONFIG ', args.config)
    print('RENDER FRAMES: ', len(scene_data['view_matrix']))
    n_rendered = 0
    ext = None
    # for i in tqdm(i_list):
    for i in tqdm(range(len(camera_labels))):
        if str(camera_labels[i]) not in cameras_subset:
            # print(f'skipping camera {camera_labels[i]}', type(camera_labels[i]))
            continue

        vm = view_matrix[i]

        if not np.isfinite(vm).all():
            # print('skip ',i)
            continue

        if image_sizes is not None and tuple(image_sizes[i]) != tuple(off_render.viewport_size):
            raise NotImplementedError()
            #off_render = OffscreenRender(viewport_size=image_sizes[i], out_buffer_location=out_buffer_location)

        scene.set_camera_view(vm)
        # scene.set_proj_matrix(proj_matrix if proj_matrix.ndim == 2 else proj_matrix[i])
        scene.set_proj_matrix(scene_data['proj_matrix'] if scene_data['proj_matrix'].ndim == 2 else scene_data['proj_matrix'][i])

        if args.mode == 'dataset':
            arrs = {'view_matrix': vm}
            for what in args.inputs:
                input_config = parse_input_string(what)
                scene.set_params(**input_config)
                frame = off_render.render(scene)

                arrs.update({what: to_numpy(frame)})

            write_data(camera_labels[i], arrs)
        elif args.mode == 'infer':
            # if ms_model is None:
            #     obsolete_input = False if 'obsolete_input' not in config else config['obsolete_input']
            #     inputs = nn_.render_input(net_args['use'], scene, off_render, parse_obsolete=obsolete_input)
            #     inputs = [torch.Tensor(x).cuda() for x in inputs]
            #     frame = nn_.infer_neural_texture(net, texture, inputs, texture_pca=args.render_desc)
            # else:
            #     frame=ms_model.infer()
            frame=model.infer()

            write_images(camera_labels[i], {'desc' if args.render_desc else 'infer': to_numpy(frame, flipv=False)}, 
                         pad_to=args.pad_to, out_extension=args.out_extension, name_only=args.cam_as_name)
        elif args.mode == 'overlay':
            frame = off_render.render(scene)
            frame_flipped = frame[::-1].copy()
            del frame

            mask = (frame_flipped[:,:,0]>1e-12)
            if ext is None:
                src_path = glob.glob(os.path.join(scene_data['config']['frames'], camera_labels[i]+'*'))[0]
                ext = os.path.splitext(src_path)[-1]

            src_image = load_image(os.path.join(scene_data['config']['frames'], 
                camera_labels[i]+ext))[:frame_flipped.shape[0], :frame_flipped.shape[1],  :frame_flipped.shape[2]]/255.
            src_image[mask] = frame_flipped[mask]

            # cv2_write(os.path.join(args.save_dir, camera_labels[i]+'.png'), src_image)
            cv2_write(os.path.join(args.save_dir, camera_labels[i]+'.' + args.out_extension), src_image)
            del src_image, mask, frame_flipped
        n_rendered += 1

        # if args.debug and n_rendered == n_max:
        #     break