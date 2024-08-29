from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404
from django.utils import timezone
import os
from django.conf import settings
import subprocess
import mimetypes

def run_command(user_id, image_path):
    project_name = user_id
    
    command = [
        'autotrain', 'dreambooth', '--train',
        '--model', 'stabilityai/stable-diffusion-xl-base-1.0',
        '--project-name', project_name,
        '--image-path', image_path,
        '--prompt', f"A photo of {user_id} wearing casual clothes and smiling.",
        '--resolution', '1024',
        '--batch-size', '1',
        '--num-steps', '1',
        '--gradient-accumulation', '4',
        '--lr', '1e-4',
        '--mixed-precision', 'fp16'
    ]
    
    try:
        subprocess.run(command, check=True)
        run_post_training_code(user_id)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def run_post_training_code(user_id):
    from diffusers import DiffusionPipeline, AutoencoderKL
    import torch

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda")
    pipe.load_lora_weights(f"{user_id}", weight_name="pytorch_lora_weights.safetensors")

    prompt = f"A portrait of {user_id} wearing a professional business suit in an professional office"
    images = pipe(prompt=prompt, num_inference_steps=25, num_images_per_prompt=3)

    save_generated_images(images, user_id)
    del vae
    del pipe
    torch.cuda.empty_cache()

def save_generated_images(images, user_id):
    generated_dir = f'media/generated/{user_id}/'
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    
    for i, img in enumerate(images['images']):
        img_path = os.path.join(generated_dir, f'generated_{i}.png')
        img.save(img_path)

def upload_view(request):
    context = {}
    if request.method == 'POST':
        username = request.POST.get('name')
        images = request.FILES.getlist('upload')

        if len(images) < 5:
            context['error'] = "Please upload at least 5 images."
            return render(request, 'home.html', context)

        # Create a directory for the user with the current date and time
        upload_time = timezone.now().strftime("%Y%m%d%H%M%S")
        user_dir = f"media/{username}_{upload_time}"
        user_id = f"{username}{upload_time}"
        
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # Save the uploaded files to the directory
        for image in images:
            file_path = os.path.join(user_dir, image.name)
            with open(file_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

        context['success'] = "Your images are successfully uploaded."
        context['message'] = "Please wait for 20-30 min to get your personalized images."
        run_command(user_id, user_dir)
        #request.session['user_id'] = user_id
        return redirect('generated_images',user_id=user_id)

    return render(request, 'home.html', context)

def generated_images(request, user_id):
    print('inside generated images')
    print(f"this is {user_id}")
    generated_dir = f'media/generated/{user_id}/'
    
    if os.path.exists(generated_dir):
        generated_images = os.listdir(generated_dir)
        generated_images = [{
            'path': os.path.join(generated_dir, img),
            'name': img,
        } for img in generated_images]
    else:
        generated_images = []

    # Adjusted image paths for HTML template
    image_paths = [os.path.join('/media/generated', user_id, f'generated_{i}.png') for i in range(3)]

    print('before return render')
    return render(request, 'generated_images.html', {
        'image_path1': image_paths[0],
        'image_path2': image_paths[1],
        'image_path3': image_paths[2]
    })


def download_image(request, user_id, image_name):
    print('inside download images')
    image_path = os.path.join('media', 'generated', user_id, image_name)
    
    if os.path.exists(image_path):
        with open(image_path, 'rb') as file:
            mime_type, _ = mimetypes.guess_type(image_path)
            response = HttpResponse(file, content_type=mime_type)
            response['Content-Disposition'] = f'attachment; filename={image_name}'
            return response
    else:
        raise Http404("Image does not exist")
