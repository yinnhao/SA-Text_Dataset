CUDA_VISIBLE_DEVICES=0 
    python demo/demo_save_det.py    --config-file configs/Bridge/TotalText/R_50_poly.yaml \
                                    --input /media/dataset1/jinlovespho/data/generated_data/drealsr/val/gt/pho_tmp_256/ \
                                    --output demo_results_boxes \
                                    --opts MODEL.WEIGHTS Bridge_tt.pth MODEL.TRANSFORMER.USE_BOX True