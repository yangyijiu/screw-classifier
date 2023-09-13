from .imports import *

def download_and_extract_dataset():
	# URL of the MVTec Screws dataset
	url = 'https://drive.google.com/uc?id=11ozVs6zByFjs9viD3VIIP6qKFgjZwv9E'
	output = 'mvtec_screws.zip'

	# Download the dataset
	print("Downloading the dataset...")
	gdown.download(url, output, quiet=False)

	# Extract the dataset
	print("Extracting the dataset...")
	with zipfile.ZipFile(output, 'r') as zip_ref:
		zip_ref.extractall('.')
	os.remove(output)  # Remove the downloaded zip file
	print("Dataset downloaded and extracted successfully!")


if __name__ == "__main__":
	download_and_extract_dataset()