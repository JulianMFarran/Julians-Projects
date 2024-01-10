import torch
from torch.nn import functional as F
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, convert_to_images
from PIL import Image

def generate_image(output_path="generated_image.png"):
    # Load pre-trained BigGAN model
    model = BigGAN.from_pretrained('biggan-deep-256')

    # Generate a random noise vector
    noise_vector = truncated_noise_sample(truncation=0.4, batch_size=1)

    # Generate random class label tensor
    class_label = torch.randint(0, model.config.num_classes, (1,))

    # Convert the class label tensor to one-hot encoding
    class_label = F.one_hot(class_label, num_classes=model.config.num_classes)

    # Convert the numpy array to a torch tensor
    noise_vector = torch.from_numpy(noise_vector)

    # Convert the input tensor to torch.float
    noise_vector = noise_vector.to(torch.float)
    class_label = class_label.to(torch.float)

    # Generate image from noise vector, class label, and truncation
    with torch.no_grad():
        output = model(noise_vector, class_label, truncation=0.4)

    # Convert the output to images
    images = convert_to_images(output)

    # Save the generated image
    images[0].save(output_path)

# Generate image
generate_image()
