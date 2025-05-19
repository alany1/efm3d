from efm3d.inference.fuse import VolumetricFusion

if __name__ == '__main__':
    import os
    # keyframes = [(0, 122), (151, 280), (310, 376), (416, 553), (577, 636), (679, 840), (864, 904)]
    # keyframes = [(0, 120), (149, 265), (311, 372), (416, 547), (573, 626), (678, 834), (863, 960)]
    keyframes = [(0, 120), (143, 266), (311, 372), (415, 546), (573, 635), (678, 836), (859, 960)]
    save_dir = "/home/exx/datasets/aria/real/kitchen_v2/vol_fusion_v2_hand_detector_combination"
    os.makedirs(save_dir, exist_ok=True)
    
    output_dir = "/home/exx/mit/efm3d/output/model_release/kitchen_v2"
    voxel_res = 0.02
    
    vol_fusion = VolumetricFusion(output_dir, voxel_res=voxel_res, device="cuda:1")
    for start, end in keyframes:
        vol_fusion.reinit()
        for i in range(start, end+1):
            vol_fusion.run_step(i)
        pred_mesh = vol_fusion.get_trimesh()
        pred_mesh.export(os.path.join(save_dir, f"mesh_{start}.ply"))
        print("done with", (start, end))
            
        
    
