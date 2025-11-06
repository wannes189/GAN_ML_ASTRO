import tensorflow as tf

import os
import time
import datetime
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from IPython import display
from sklearn.model_selection import train_test_split

OUTPUT_CHANNELS = 1
LAMBDA = 100
RUN = 20

# Define paths
train_path = r"/data/gent/472/vsc47220/train"
test_path = r"/data/gent/472/vsc47220/test"
#train_path = r"C:/Users/Wannes/UGent/Thesis/January/transformedv2/test"
#test_path = r"C:/Users/Wannes/UGent/Thesis/January/transformedv2/test"

# Function to pair input and target files in a given directory
def get_file_pairs(directory):
    file_pairs = []
    galaxies = sorted(os.listdir(directory))
    for galaxy_folder in galaxies:
        galaxy_path = os.path.join(directory, galaxy_folder)
        if os.path.isdir(galaxy_path):
            galaxy_id = galaxy_folder
            input_files = sorted([f for f in os.listdir(galaxy_path) if f.endswith("_scaled.fits")])
        
            for input_file in input_files:
                input_path = os.path.join(galaxy_path, input_file)

                stem_parts = input_file.replace(".fits", "").split("_")
                if "GALEX"  not in stem_parts and "NUV" not in stem_parts and "FUV" not in stem_parts:
                    observer = stem_parts[1]  # e.g. O1
                    position = "_".join(stem_parts[2:-1])  # e.g. 0_0
                    if position != "0_0":
                        continue
                    target_file = f"{galaxy_id}_{observer}_GALEX_NUV_{position}_scaled.fits"
                    target_path = os.path.join(galaxy_path, target_file)
                    if os.path.exists(target_path):  
                        file_pairs.append((input_path, target_path))
    return file_pairs

# Collect file pairs for train and test sets
train_pairs = get_file_pairs(train_path)
test_pairs = get_file_pairs(test_path)

# Helper function to load input and target FITS files
def load_fits_pair(input_file, target_file):
    # Handle tf.Tensor or str
    if isinstance(input_file, tf.Tensor):
        input_file = input_file.numpy().decode("utf-8")
    if isinstance(target_file, tf.Tensor):
        target_file = target_file.numpy().decode("utf-8")

    with fits.open(input_file) as hdul:
        input_data = hdul[0].data

    with fits.open(target_file) as hdul:
        target_data = hdul[0].data

    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    target_data = tf.convert_to_tensor(target_data, dtype=tf.float32)

    return input_data, target_data

# Function wrapper for tf.py_function
def load_data(input_file, target_file):
    return tf.py_function(func=load_fits_pair, inp=[input_file, target_file], Tout=(tf.float32, tf.float32))

def load_data_for_ts(input_path, target_path):
    input_image, target_image = load_fits_pair(input_path, target_path)  # your function to load FITS as tensor
    return input_path, input_image, target_image    

# Extract all galaxy IDs in test set
all_test_ids = sorted(os.listdir(test_path))
galaxy_ids = [g for g in all_test_ids if os.path.isdir(os.path.join(test_path, g))]

# Split galaxy IDs into val/test
val_ids, test_ids = train_test_split(galaxy_ids, test_size=0.5, random_state=42)

# Filter file pairs by galaxy ID
def filter_pairs_by_ids(pairs, id_list):
    return [pair for pair in pairs if os.path.basename(pair[0]).split("_")[0] in id_list]

val_pairs = filter_pairs_by_ids(test_pairs, val_ids)
final_test_pairs = filter_pairs_by_ids(test_pairs, test_ids)

input_paths = [pair[0] for pair in final_test_pairs]
target_paths = [pair[1] for pair in final_test_pairs]

# Dataset of paths
paths_ds = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))

# Save the galaxy IDs to txt files
with open("val_ids.txt", "w") as f:
    for gid in sorted(val_ids):
        f.write(gid + "\n")

with open("test_ids.txt", "w") as f:
    for gid in sorted(test_ids):
        f.write(gid + "\n")

# Create TensorFlow datasets for train and test
train_dataset = tf.data.Dataset.from_tensor_slices((
    [pair[0] for pair in train_pairs], [pair[1] for pair in train_pairs]
))
train_dataset = train_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000)  # Shuffle training data
train_dataset = train_dataset.batch(batch_size=32)  # Adjust batch size as needed

# Validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((
    [pair[0] for pair in val_pairs], [pair[1] for pair in val_pairs]
))
val_dataset = val_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size=32)

# # Final test dataset
# test_dataset = tf.data.Dataset.from_tensor_slices((
#     [pair[0] for pair in final_test_pairs], [pair[1] for pair in final_test_pairs]
# ))
# test_dataset = test_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(batch_size=32)

test_dataset_with_paths = paths_ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset_with_paths = test_dataset_with_paths.batch(batch_size=32)


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[128, 128, 6])  # Change input shape to (128, 128, 6)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 64)
        downsample(128, 4),  # (batch_size, 32, 32, 128)
        downsample(256, 4),  # (batch_size, 16, 16, 256)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(256, 4),  # (batch_size, 16, 16, 512)
        upsample(128, 4),  # (batch_size, 32, 32, 256)
        upsample(64, 4),   # (batch_size, 64, 64, 128)
        upsample(1, 4),    # (batch_size, 128, 128, 1) - final output with 1 channel
    ]

    initializer = tf.keras.initializers.Zeros()
    last = tf.keras.layers.Conv2DTranspose(
        1, 4, strides=2, padding='same',
        kernel_initializer=initializer,
        bias_initializer=initializer,  # Ensures bias starts at 0
        activation='tanh'
    ) # (batch_size, 128, 128, 1)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    # Compute GAN loss
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Compute L1 loss (make sure gen_output is squeezed to remove the last channel dimension)
    l1_loss = tf.reduce_mean(tf.abs(target - tf.squeeze(gen_output, axis=-1)))  # Remove the last dimension

    # Total generator loss
    total_gen_loss = gan_loss + LAMBDA * l1_loss
    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[128, 128, 6], name='input_image')   # 6-channel input
    tar = tf.keras.layers.Input(shape=[128, 128, 1], name='target_image')  # 1-channel target

    # Concatenate: (128, 128, 7)
    x = tf.keras.layers.concatenate([inp, tar])

    # Downsample blocks
    down1 = downsample(64, 4, apply_batchnorm=False)(x)   # (64, 64, 64)
    down2 = downsample(128, 4)(down1)                     # (32, 32, 128)
    down3 = downsample(256, 4)(down2)                     # (16, 16, 256)

    # Extra conv to reduce to 13x13
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)    # (18, 18, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (15, 15, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (17, 17, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (14, 14, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

log_dir=f"logs{RUN}/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def PSNR(gen_output, target): 
    mse = np.mean((target - gen_output) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr

def calculate_uqi(x, y):
    """
    Calculate the Universal Quality Index (UQI) between two images.
    
    Parameters:
    x (numpy.ndarray): Original image as a 2D (or flattened) array.
    y (numpy.ndarray): Distorted/generated image as a 2D (or flattened) array.
    
    Returns:
    float: UQI value.
    """

    # Convert tensors to numpy arrays if needed
    if hasattr(x, 'numpy'):
        x = x.numpy()
    if hasattr(y, 'numpy'):
        y = y.numpy()
    
    # Squeeze any extra singleton dimensions
    x = np.squeeze(x)
    y = np.squeeze(y)

    # Check if they are now matching
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch after squeezing: {x.shape} vs {y.shape}")


    x = x.flatten()
    y = y.flatten()
    N = len(x)
    
    # Means
    bx = np.mean(x)
    by = np.mean(y)
    
    # Standard deviations
    rx = np.std(x)
    ry = np.std(y)
    
    # Covariance
    rxy = np.mean((x - bx) * (y - by))
    
    # Avoid division by zero
    denominator = (bx**2 + by**2) * (rx**2 + ry**2)
    if denominator == 0:
        return 0.0
    
    # UQI formula
    uqi = (4 * rxy * bx * by) / denominator
    
    return uqi

@tf.function
def tf_uqi(x, y):
    def uqi_fn(x_np, y_np):
        x_np = tf.reshape(x_np, [-1])
        y_np = tf.reshape(y_np, [-1])
        bx = np.mean(x_np)
        by = np.mean(y_np)
        rx = np.std(x_np)
        ry = np.std(y_np)
        rxy = np.mean((x_np - bx) * (y_np - by))
        denominator = (bx**2 + by**2) * (rx**2 + ry**2)
        if denominator == 0:
            return np.float32(0.0)
        uqi = (4 * rxy * bx * by) / denominator
        return np.float32(uqi)
    
    uqi = tf.py_function(uqi_fn, [x, y], tf.float32)
    return uqi

@tf.function
def PSNR(x, y):
    if x.shape[-1] != y.shape[-1]:
        y = tf.expand_dims(y, axis=-1)
    return tf.image.psnr(x, y, max_val=1.0)

@tf.function
def SSIM(x, y):
    if x.shape[-1] != y.shape[-1]:
        y = tf.expand_dims(y, axis=-1)
    return tf.image.ssim(x, y, max_val=1.0)

@tf.function
def MSE(x, y):
    if x.shape[-1] != y.shape[-1]:
        y = tf.expand_dims(y, axis=-1)
    return tf.reduce_mean(tf.square(x - y))

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # Metrics (reduce over batch)
    psnr = tf.reduce_mean(PSNR(gen_output, target))
    ssim_value = tf.reduce_mean(SSIM(gen_output, target))
    mse = tf.reduce_mean(MSE(gen_output, target))
    uqi = tf.reduce_mean(tf_uqi(target, gen_output))  # assuming tf_uqi is batch-compatible

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)
        tf.summary.scalar('PSNR', psnr, step=step // 1000)
        tf.summary.scalar('SSIM', ssim_value, step=step // 1000)
        tf.summary.scalar('MSE', mse, step=step // 1000)
        tf.summary.scalar('UQI', uqi, step=step // 1000)

@tf.function
def validation_step(input_image, target, step):
    # Generate the output (no training)
    gen_output = generator(input_image, training=False)

    # Metrics
    psnr = PSNR(gen_output, target)
    ssim_value = SSIM(gen_output, target)
    mse = MSE(gen_output, target)
    uqi = tf_uqi(target, gen_output)

    # Log validation metrics
    with summary_writer.as_default():
        tf.summary.scalar('val_PSNR', psnr, step=step)
        tf.summary.scalar('val_SSIM', ssim_value, step=step)
        tf.summary.scalar('val_MSE', mse, step=step)
        tf.summary.scalar('val_UQI', uqi, step=step)

def run_validation(val_ds, step):
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_uqi = 0.0
    total_samples = 0

    for val_input, val_target in val_ds:
        gen_output = generator(val_input, training=False)

        # PSNR, SSIM, and MSE return per-image values
        psnr_values = PSNR(gen_output, val_target)  # shape (batch_size,)
        ssim_values = SSIM(gen_output, val_target)  # shape (batch_size,)
        mse_values = MSE(gen_output, val_target)    # shape (batch_size,)

        batch_size = tf.shape(psnr_values)[0]

        total_psnr += tf.reduce_sum(psnr_values)
        total_ssim += tf.reduce_sum(ssim_values)
        total_mse += tf.reduce_sum(mse_values)

        # For UQI, loop over the batch since it's calculated per image
        for i in range(batch_size):
            total_uqi += calculate_uqi(val_target[i], gen_output[i])

        total_samples += batch_size

    avg_psnr = total_psnr / tf.cast(total_samples, tf.float32)
    avg_ssim = total_ssim / tf.cast(total_samples, tf.float32)
    avg_mse = total_mse / tf.cast(total_samples, tf.float32)
    avg_uqi = total_uqi / tf.cast(total_samples, tf.float32)

    with summary_writer.as_default():
        tf.summary.scalar('val_PSNR', avg_psnr, step=step)
        tf.summary.scalar('val_SSIM', avg_ssim, step=step)
        tf.summary.scalar('val_MSE', avg_mse, step=step)
        tf.summary.scalar('val_UQI', avg_uqi, step=step)

    print(f"\nValidation at step {step}: PSNR={avg_psnr}, SSIM={avg_ssim}, MSE={avg_mse}, UQI={avg_uqi}")


def save_test_predictions(model, test_file_pairs, output_dir='test_predictions_center'):
    
    os.makedirs(output_dir, exist_ok=True)

    for input_path, target_path in test_file_pairs:
        # Load the input image only
        input_image = load_fits_pair(input_path, target_path)[0]  # (H, W, C)

        # Add batch dimension: (1, H, W, C)
        input_batch = tf.expand_dims(input_image, axis=0)

        # Predict
        prediction = model(input_batch, training=False).numpy()[0]  # Remove batch dimension

        # Remove channel dimension if present
        if prediction.ndim == 3 and prediction.shape[-1] == 1:
            prediction = prediction[..., 0]

        # Construct output name
        file_name = os.path.basename(input_path)
        stem_parts = file_name.replace(".fits", "").split("_")
        galaxy_id = stem_parts[0]
        observer = stem_parts[1]
        position = "_".join(stem_parts[2:-1])  # everything after observer except 'scaled'

        output_name = f"{galaxy_id}_{observer}_GALEX_NUV_{position}_predicted_{RUN}.fits"
        output_path = os.path.join(output_dir, output_name)

        # Save prediction as FITS
        fits.writeto(output_path, prediction, overwrite=True)

def fit(train_ds, val_ds, test_ds, steps, run=RUN): 
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()
            run_validation(val_ds, step)
            

            print(f"Step: {step//1000}k")

        # Perform a training step
        train_step(input_image, target, step)

        # Progress indicator
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)
    print("Saving final predictions...")
    save_test_predictions(generator, test_ds)


fit(train_dataset, val_dataset, final_test_pairs, steps=50000)