import gdown

# Google Drive file ID
file_id = "1yGnMCPgNzEID-iFqFKw9mzZl_zF-W68f"
url = f"https://drive.google.com/uc?id={1yGnMCPgNzEID-iFqFKw9mzZl_zF-W68f}"

# Download model
gdown.download(url, "Final_predication_SoftVoting.pkl", quiet=False)
print("✅ Model downloaded successfully!")
               
