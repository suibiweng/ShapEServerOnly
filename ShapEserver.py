import torch 
import subprocess 
from  flask import Flask, jsonify, request ,send_from_directory 
from flask_limiter import Limiter 
from flask_limiter.util import get_remote_address 
import requests 
import io 
import base64 
import json 
from shap_e.diffusion.sample import sample_latents 
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config 
from shap_e.models.download import load_model, load_config 
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images,gif_widget 
from shap_e.util.notebooks import decode_latent_mesh 
import os
import pymeshlab
import zipfile

app = Flask(__name__) 
limiter = Limiter( 
    get_remote_address, 
    app=app, 
) 
 
device = torch.device('cuda') 
xm = load_model('transmitter', device=device) 
model = load_model('text300M', device=device) 
diffusion = diffusion_from_config(load_config('diffusion')) 


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    directory_path = "./"  # Adjust as needed
    file_path = os.path.join(directory_path, filename)

    print(f"Requested file: {file_path}")  # Log the file path

    if not os.path.exists(file_path):
        print("File not found!")  # Log if the file does not exist
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(directory_path, filename, as_attachment=True)


@app.route("/generate", methods=["POST"])
def generate_model():
    data = request.json
    urid = data.get("URID")
    prompt = data.get("prompt")
    filename = data.get("filename")

    if not urid or not prompt or not filename:
        return jsonify({"error": "Missing URID, prompt, or filename"}), 400



    # zip_path = f"./{urid}.zip"
    # final_zip_path = f"output/{filename}.zip"
    result = generate(prompt, 24, 60,'obj',urid)

    # Rename the zip file to the provided filename

    # os.rename(zip_path, final_zip_path)

    return jsonify({"message": result, "zip_file": "final_zip_path"})

 
@app.route('/data', methods=['POST']) 
def data(): 
    json_data = request.get_json() 
    prompt = json_data['prompt'] 
    cfg = json_data['cfg'] 
    steps = json_data['steps'] 
    if int(steps)>64: 
        steps=64 
    fileFormat = json_data['fileFormat'] 
    fileFormat = fileFormat.lower() 
    return generate(prompt, cfg, steps,fileFormat) 
 
def generate(prompt,cfg,steps,fileFormat,filename): 
 
    latents = sample_latents( 
        batch_size=1, 
        model=model, 
        diffusion=diffusion, 
        guidance_scale=int(cfg), 
        model_kwargs=dict(texts=[prompt]), 
        progress=True, 
        clip_denoised=True, 
        use_fp16=True, 
        use_karras=True, 
        karras_steps=int(steps), 
        sigma_min=1E-3, 
        sigma_max=160, 
        s_churn=0, 
    ) 
 
    for i, latent in enumerate(latents): 
        t = decode_latent_mesh(xm, latent).tri_mesh() 
        with open(f'plyFile.ply', 'wb') as f: 
            t.write_ply(f) 


    export_file('plyFile.ply',f'{filename}.{fileFormat}',fileFormat,filename)
 
    # subprocess.run(f"python3 convert.py {fileFormat}", shell=True) 
 
    # with open(f"outputFile.{fileFormat}", "rb") as f: 
    #     b64 = base64.b64encode(f.read()).decode("utf-8") 
    #     return b64 
    
def export_file(input_file, output_file, output_format,urlid):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_file)

    # Check if vertex colors exist before proceeding
    if not ms.current_mesh().has_vertex_color():
        print("Warning: Mesh does not have vertex colors!")

    # # Apply rotation transformations
    # ms.apply_filter('compute_matrix_from_rotation', rotaxis='X', angle=-90.0)
    # ms.apply_filter('compute_matrix_from_rotation', rotaxis='Y', angle=180.0)

    # Define export options to preserve vertex colors
    export_options = {
        "save_vertex_color": True,  # Ensure vertex colors are saved
        "save_face_color": True,    # Preserve face colors if available
        "save_wedge_texcoord": True # Ensures texture coordinates are saved when applicable
    }

    # Export based on format
    if output_format in ["ply", "glb", "gltf"]:
        ms.save_current_mesh(output_file, **export_options)
        print(f"Exported {output_file} with colors.")
    elif output_format == "obj":
        print("Warning: OBJ format does not support vertex colors natively.")
        ms.save_current_mesh(output_file, save_vertex_color=True,)  # OBJ only stores per-face colors
    elif output_format == "fbx":
        ms.save_current_mesh(output_file, **export_options)
        print(f"Exported {output_file} with colors (FBX may require extra setup).")
    elif output_format == "blend":
        print("PyMeshLab does not support exporting to .blend format.")
    else:
        print("Invalid output file format. Please choose from ply, fbx, obj, gltf, or glb.")
        
        
    zip_file = f"{urlid}_ShapE.zip"
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, os.path.basename(output_file))

 
if __name__ == '__main__': 
    app.run(debug=False, host = '0.0.0.0') 