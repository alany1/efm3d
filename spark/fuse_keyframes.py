from efm3d.inference.fuse import VolumetricFusion

if __name__ == '__main__':
    import os
    keyframes = [(0, 120), (152, 267), (310, 368), (417, 551), (577, 636), (680, 838), (868, 904)]
    save_dir = "/home/exx/Downloads/vol_fusion_test"
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
            
        
    
