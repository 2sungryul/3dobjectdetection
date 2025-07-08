import argparse
import glob
from pathlib import Path
import open3d
from visual_utils import open3d_vis_utils as V
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def model_inference(cfg_file, data_path, ckpt_path):
    #ext = '.bin'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    
    demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(data_path))
    print(f'Total number of samples: \t{len(demo_dataset)}')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    pcd = []
    pred_dict = {}
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pcd = torch.Tensor.tolist(data_dict['points'][:, 1:4])
            pred_dict['pred_boxes'] = torch.Tensor.tolist(pred_dicts[0]['pred_boxes'])
            pred_dict['pred_scores'] = torch.Tensor.tolist(pred_dicts[0]['pred_scores'])
            pred_dict['pred_labels'] = torch.Tensor.tolist(pred_dicts[0]['pred_labels'])

    return pred_dict, pcd

if __name__ == '__main__':

    cfg_file = "pointpillar.yaml"
    #cfg_file = "pointrcnn.yaml"
    #cfg_file = "second.yaml"
    #cfg_file = "voxel_rcnn_car.yaml"
    #cfg_file = "pv_rcnn.yaml"

    data_path = "000000.bin"
    #data_path = "000008.bin"
        
    ckpt_path = "pointpillar_7728.pth"
    #ckpt_path = "pointrcnn_7870.pth"
    #ckpt_path = "second_7862.pth"
    #ckpt_path = "voxel_rcnn_car_84.54.pth"
    #ckpt_path = "pv_rcnn_8369.pth"
    
    pred_dict,pcd = model_inference(cfg_file=cfg_file, data_path=data_path, ckpt_path=ckpt_path)
   
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name="kitti")
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(pcd)
    pts.colors = open3d.utility.Vector3dVector(np.ones((len(pcd), 3)))
    vis.add_geometry(pts)

    box_color = [[0,1,0],[0,1,1],[1,1,0],[1,1,1]]

    for i in range(len(pred_dict['pred_boxes'])):
            center = pred_dict['pred_boxes'][i][0:3]
            lwh = pred_dict['pred_boxes'][i][3:6]
            axis_angles = np.array([0, 0, pred_dict['pred_boxes'][i][6]])
            rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
            box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
            box3d.color = box_color[pred_dict['pred_labels'][i]-1] 
            vis.add_geometry(box3d)           
        
    vis.run()
    vis.destroy_window()






        
    
