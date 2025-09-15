from transformers import GPT2Model, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('../pretrain/GPT2')
model = GPT2Model.from_pretrained('../pretrain/GPT2')
print("GPT2 = {}".format(model))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define prompts with names
text_prompts = {
    "ETT": ("The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. "
            "This dataset consists of 2 years data from two separated counties in China. "
            "There is a time-series data segment. Your task is to forecast carefully the future steps."),
    "Tra": ("Traffic is a collection of hourly data from California Department of Transportation. "
            "This dataset describes the road occupancy rates measured by different sensors on San Francisco Bay area freeways. "
            "There is a time-series data segment. Your task is to forecast carefully the future steps."),
    "Wea": ("Weather is recorded every 10 minutes for the 2020 whole year. "
            "This dataset contains 21 meteorological indicators, such as air temperature, humidity, etc. "
            "There is a time-series data segment. Your task is to forecast carefully the future steps."),
    "Ele": ("Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. "
            "Different electrical quantities and some sub-metering values are available. "
            "This dataset contains 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months). "
            "There is a time-series data segment. Your task is to forecast carefully the future steps.")
}

encoded_prompts = {}

# Encode each prompt
for name, text in text_prompts.items():
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get mean-pooled embeddings
    embedding = outputs.last_hidden_state.mean(dim=1, keepdim=False)[0]

    # Store embedding in dictionary with corresponding name
    encoded_prompts[name] = embedding
    print(f"Processed prompt: {name}, Embedding shape: {embedding.shape}")

# Save embeddings for future use
torch.save(encoded_prompts, 'text_prompts_GPT2.pt')
print("Saved all prompt embeddings.")

# Example of loading and accessing a specific embedding
loaded_prompts = torch.load('text_prompts_GPT2.pt')
print("Loaded embedding for 'Weather':", loaded_prompts["Wea"].shape)
