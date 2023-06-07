import os
import argparse
import pymeshlab
import time

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="Enter the models folder path")
args = vars(ap.parse_args())

modelsFolder = args["folder"]

modelPaths = [os.path.join(modelsFolder, i) for i in sorted(os.listdir(modelsFolder)) if i.endswith('obj')]

outputFolder = os.path.join(os.path.dirname(modelsFolder), os.path.basename(modelsFolder) + "_processed")

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

counter = 0
start = time.time()

for modelPath in modelPaths:

    process_start = time.time()
    counter += 1	
    model = os.path.basename(modelPath)
    print("\nProcessing Model ", counter, " : ", model)

    savePath = os.path.join(outputFolder, model)

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(modelPath)
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
        ms.transfer_attributes_to_texture_per_vertex(textname=model[:-4])
        print("Texture defragmentation")
        ms.apply_texmap_defragmentation()
        ms.save_current_mesh(savePath, save_textures=True, texture_quality=-1)
		
        process_time = time.time() - process_start
        print('Processed and saved in ', format(process_time, '.4f'), "Seconds")
		
    except Exception as e:
        print("Failed to process : ", model, " > Error : ", e)
        pass

total_time = time.time() - start
print("\nBatch process finished in ", time.strftime("%Hh %Mm %Ss ",time.gmtime(total_time)), " !!!\n")