from flask import Flask, request, jsonify, send_from_directory
import requests
import base64
import os
import shutil
import pymeshlab as ml

app = Flask(__name__)

class SHAPERuntime:
    def __init__(self):
        self.steps = 64
        self.cfg = 20
        self.directory_path = "./output"  # Do not use / at the end
        self.format = "FBX"
        self.text_to_mesh_id = "nejnwmcwvhcax9"
        self.invoice = "20067068964066"
        self.model_name = ""

    def start(self):
        self.verify(f"https://{self.text_to_mesh_id}-5000.proxy.runpod.net/verify", f'{{"invoice":"{self.invoice}"}}')

    def send_prompt(self, user_id, prompt, model_id):
        self.model_name = model_id
        url = f"https://{self.text_to_mesh_id}-5000.proxy.runpod.net/data"
        body = {
            "prompt": prompt,
            "steps": self.steps,
            "cfg": self.cfg,
            "invoice": self.invoice,
            "fileFormat": self.format
        }
        response = requests.post(url, json=body)

        if response.status_code == 200:
            model_data = base64.b64decode(response.text)
            dpath = f"{self.directory_path}/{model_id}"
            os.makedirs(dpath, exist_ok=True)

            fbx_path = f"{dpath}/{model_id}.fbx"
            obj_path = f"{dpath}/{model_id}.obj"

            with open(fbx_path, "wb") as model_file:
                model_file.write(model_data)

            fbx_to_obj(fbx_path, obj_path)

            # Zip the directory
            zip_path = f"{self.directory_path}/{model_id}.zip"
            shutil.make_archive(dpath, 'zip', dpath)

            return f"Model generated successfully. Files saved and zipped at {zip_path}"
        else:
            return f"Error: {response.status_code}, {response.text}"

    def verify(self, url, body_json_string):
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=body_json_string.encode('utf-8'), headers=headers)
        if response.status_code == 200 and response.text != "Not Verified":
            print(f"Invoice verified. Remaining objects: {response.text}")
        else:
            print(f"Verification failed: {response.text}")


def fbx_to_obj(input_file, output_file):
    ms = ml.MeshSet()
    ms.load_new_mesh(input_file)
    ms.save_current_mesh(output_file)


@app.route("/generate", methods=["POST"])
def generate_model():
    data = request.json
    urid = data.get("URID")
    prompt = data.get("prompt")
    filename = data.get("filename")

    if not urid or not prompt or not filename:
        return jsonify({"error": "Missing URID, prompt, or filename"}), 400

    shaperuntime = SHAPERuntime()
    shaperuntime.start()

    result = shaperuntime.send_prompt("", prompt, urid)

    # Rename the zip file to the provided filename
    zip_path = f"{shaperuntime.directory_path}/{urid}.zip"
    final_zip_path = f"{shaperuntime.directory_path}/{filename}.zip"
    os.rename(zip_path, final_zip_path)

    return jsonify({"message": result, "zip_file": final_zip_path})

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    directory_path = "./output"  # Adjust as needed
    file_path = os.path.join(directory_path, filename)

    print(f"Requested file: {file_path}")  # Log the file path

    if not os.path.exists(file_path):
        print("File not found!")  # Log if the file does not exist
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(directory_path, filename, as_attachment=True)

    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
