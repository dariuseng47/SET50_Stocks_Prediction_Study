import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Import configurations from the parent or specific config path
# For 3rd Study, we will need to ensure pathing is correct.
import sys
# Assuming we run from project root, we might need to adjust this.

def calculate_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Calculates the gradient penalty for WGAN-GP.
    """
    batch_size = tf.shape(real_samples)[0]
    # Get the interpolated samples
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_samples - real_samples
    interpolated = real_samples + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # Get the discriminator output for the interpolated samples
        pred = discriminator(interpolated, training=True)

    # Calculate the gradients w.r.t. the interpolated samples
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # Calculate the norm of the gradients
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

class WGANGPTrainerV1:
    def __init__(self, generator, discriminator, 
                 g_optimizer=None, d_optimizer=None, 
                 gp_weight=10.0, l1_lambda=100.0, n_critic=5):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer or Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = d_optimizer or Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.gp_weight = gp_weight
        self.l1_lambda = l1_lambda
        self.n_critic = n_critic

    @tf.function
    def train_step_discriminator(self, real_images, condition_images, noise):
        """
        One training step for the Discriminator (Critic).
        """
        with tf.GradientTape() as tape:
            # 1. Generate fake images
            fake_images = self.generator([noise, condition_images], training=True)
            
            # 2. Get logits for real and fake
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            # 3. WGAN Loss: E[fake] - E[real]
            d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            
            # 4. Gradient Penalty
            gp = calculate_gradient_penalty(self.discriminator, real_images, fake_images)
            
            # 5. Total Loss
            d_loss = d_cost + gp * self.gp_weight

        # 6. Apply gradients
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss

    @tf.function
    def train_step_generator(self, condition_images, target_price, noise):
        """
        One training step for the Generator with L1 Loss.
        """
        with tf.GradientTape() as tape:
            # 1. Generate fake sequence
            fake_sequences = self.generator([noise, condition_images], training=True)
            
            # 2. Get discriminator's opinion
            gen_logits = self.discriminator(fake_sequences, training=True)
            
            # 3. Adversarial Loss: -E[fake]
            g_adv_loss = -tf.reduce_mean(gen_logits)
            
            # 4. L1 Loss (Next Day Prediction vs Real Target)
            # Assuming output sequence length is 1: (Batch, 1, Features)
            pred_val = fake_sequences[:, 0, 0] 
            target_val = tf.cast(target_price, tf.float32)
            l1_loss = tf.reduce_mean(tf.abs(pred_val - target_val))
            
            # 5. Total Generator Loss
            g_loss = g_adv_loss + (self.l1_lambda * l1_loss)

        # 6. Apply gradients
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, l1_loss

    def train(self, dataset, epochs, noise_dim, batch_size, save_path=None):
        """
        Main training loop.
        """
        best_mae = float('inf')
        history = {'d_loss': [], 'g_loss': [], 'mae': []}
        
        for epoch in range(epochs):
            epoch_mae_list = []
            epoch_d_loss_list = []
            epoch_g_loss_list = []
            
            for batch_idx, (cond_batch, target_batch) in enumerate(dataset):
                # Target needs to be (Batch, 1, 1) for Discriminator
                real_target_img = tf.reshape(tf.cast(target_batch, tf.float32), [batch_size, 1, 1])
                cond_batch = tf.cast(cond_batch, tf.float32)
                
                # A. Train Discriminator
                noise = tf.random.normal([batch_size, noise_dim])
                d_loss = self.train_step_discriminator(real_target_img, cond_batch, noise)
                epoch_d_loss_list.append(float(d_loss))

                # B. Train Generator (every n_critic steps)
                if (batch_idx + 1) % self.n_critic == 0:
                    noise = tf.random.normal([batch_size, noise_dim])
                    g_loss, mae_loss = self.train_step_generator(cond_batch, target_batch, noise)
                    epoch_g_loss_list.append(float(g_loss))
                    epoch_mae_list.append(float(mae_loss))

            # Epoch summary
            avg_mae = np.mean(epoch_mae_list) if epoch_mae_list else float('inf')
            avg_d_loss = np.mean(epoch_d_loss_list) if epoch_d_loss_list else 0
            avg_g_loss = np.mean(epoch_g_loss_list) if epoch_g_loss_list else 0
            
            history['d_loss'].append(avg_d_loss)
            history['g_loss'].append(avg_g_loss)
            history['mae'].append(avg_mae)
            
            # Save best model
            if avg_mae < best_mae:
                best_mae = avg_mae
                if save_path:
                    self.generator.save(save_path)
            
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | MAE: {avg_mae:.4f} (Best: {best_mae:.4f})")

        return best_mae, history
