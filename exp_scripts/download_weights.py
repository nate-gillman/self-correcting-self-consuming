import gdown
import os
import zipfile

def download_and_extract_zip_gdown(url, destination):
    """
    Downloads a ZIP file from Google Drive using gdown, unzips it to the specified destination,
    ignoring macOS metadata files and directories, and then deletes the original zip file.

    Args:
    - url: The Google Drive URL of the file to download.
    - destination: The directory where the ZIP file should be extracted.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)
    
    # Generate the path for the temporary download
    zip_filename = os.path.join(destination, "temp_download.zip")

    print("Downloading file...")
    # Download the file using gdown
    gdown.download(url=url, output=zip_filename, fuzzy=True)
    
    print("Extracting files...")
    # Extract the ZIP file, ignoring unwanted macOS-specific files and folders
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # Filter out the files and directories we don't want
        good_files = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX') and not os.path.basename(f).startswith('._')]
        # Extract only the filtered files
        for file in good_files:
            zip_ref.extract(file, destination)
    
    print("Cleaning up...")
    # Delete the ZIP file
    os.remove(zip_filename)
    print("Done!")




if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1MdUEazbvY1vxrBw_mRZ_XFWlj6vUmLDS/view?usp=drive_link"
    destination = "exp_outputs/dataset_0064"  # Change this to your desired path
    download_and_extract_zip_gdown(url, destination)