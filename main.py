import transformers as tr
import torch

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

###########
# Helpers #
###########

def get_probabilities(model, tokenized, past_key_values, key):
    ''' 
    Returns probabilities generated over the space of next tokens.

    model: a transformer
    tokenized: tensor of token ids of shape (1, # of tokens) if no cached activations, or (1, 1) if cached activations
    past_key_values: dictionary containing cached activations from previous run
    key: 'amateur' or 'expert', so that we can access the correct activations from the dictionary past_key_values
    '''
    with torch.no_grad():
        past_values = past_key_values[key]
        if past_values is None:
            outputs = model(tokenized)
        else:
            outputs = model(tokenized, past_key_values = past_values, use_cache = True)
    past_key_values[key] = outputs.past_key_values
    logits = outputs.logits.squeeze(0)
    probabilities = torch.softmax(logits, dim = 1)
    # we only want the logits for the next token
    out = probabilities[-1]
    return out

def get_cd_objective(amateur, expert, prompt, past_key_values):
    ''' 
    Gets the cd objective over the space of next possible tokens, and returns it in the form of a 1d tensor
    
    amateur: amateur model, a transformer
    expert: expert model, a transformer
    prompt: tensor of token ids of shape (1, # of tokens) if no cached activations, or (1, 1) if cached activations
    past_key_values: dictionary containing cached activations from previous run
    '''
    adaptive_plausibility_constraint = 0.1
    amateur_probabilities = get_probabilities(amateur, prompt, past_key_values, 'amateur')
    expert_probabilities = get_probabilities(expert, prompt, past_key_values, 'expert')
    max_prob = torch.max(expert_probabilities)
    plausibility_mask = expert_probabilities < adaptive_plausibility_constraint * max_prob
    # sets all unplausible options to have 0 probability. This means the objective for this token will be negative infinity
    expert_probabilities[plausibility_mask] = 0
    cd_objectives = torch.log(expert_probabilities/amateur_probabilities)
    return cd_objectives

def generate_next_token(amateur, expert, prompt, past_key_values):
    ''' 
    Returns the next token, an integer.

    amateur: amateur model, a transformer
    expert: expert model, a transformer
    prompt: tensor of token ids of shape (1, # of tokens) if no cached activations, or (1, 1) if cached activations
    past_key_values: dictionary containing cached activations from previous run
    '''
    cd_objective = get_cd_objective(amateur, expert, prompt, past_key_values)
    next_token = torch.argmax(cd_objective)
    return next_token


########################
# Contrastive Decoding #
########################

def contrastive_generation(amateur, expert, prompt, max_tokens) -> str:
    ''' 
    Uses contrastive decoding to generate tokens until the max_tokens number has been reached.
    To clarify, max_tokens is the number of total tokens, not the number of tokens generated.
    Optimized using KV caching to avoid having to repeatedly calculate the same attention scores. 

    Amateur, Expert: huggingface models, not paths
    Prompt: 1D tensor of ids (so that we can do tokenization outside of the function)
    Max_tokens: an int specifiying when we should stop generating text.
    '''
    output = list(prompt[0])
    past_key_values = {'amateur': None, 'expert': None}
    next_token = None
    print('Starting to generate text!')
    # Note prompt has shape (1, # of tokens)
    for i in range(max_tokens - len(prompt[0])):
        # if we haven't cached any activations yet, use the whole prompt
        # else, just use the next new token
        if next_token is None:
            next_token = generate_next_token(amateur, expert, prompt, past_key_values)
        else:
            next_token_tensor = torch.tensor([[next_token]])
            next_token = generate_next_token(amateur, expert, next_token_tensor, past_key_values)
        output.append(next_token)
        # if it wants to terminate early, then just return output
        if next_token == tokenizer.eos_token_id:
            return tokenizer.decode(output)
        if i % 10 == 0:
            print(f'Generated {i} tokens so far!')
    return tokenizer.decode(output)



expert = tr.AutoModelForCausalLM.from_pretrained(expert_path)
amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path)

tokenized = torch.tensor(tokenizer(prompt)['input_ids']).unsqueeze(0)

output_string = contrastive_generation(amateur, expert, tokenized, 500)
print(output_string)
