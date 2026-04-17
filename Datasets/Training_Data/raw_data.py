from datasets import load_dataset

# Load the SROIE dataset
sroie_dataset = load_dataset("rth/sroie-2019-v2", split="train")
# Load the katanaml dataset
katanaml_dataset = load_dataset("katanaml-org/invoices-donut-data-v1", split="train")

# Let's look at the first row
first_record = sroie_dataset[0]

# Ignore the massive 'objects' dictionary entirely. 
# Extract only the 'entities'
clean_headers = first_record["objects"]["entities"]

print(clean_headers)
# Output: {'company': 'BOOK TA .K SDN BHD', 'date': '25/12/2018', 'address': 'NO.53 JALAN SAGU...', 'total': '9.00'}