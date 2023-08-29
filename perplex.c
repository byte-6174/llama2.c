#define TESTING
#include "run.c"

void assert_eq(int a, int b) {
    if (a != b) {
        printf("Assertion failed: %d != %d\n", a, b);
        exit(EXIT_FAILURE);
    }
}
float calculate_nll_for_chunk(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, int* prompt_tokens, int num_prompt_tokens, int steps, float nll) {
    int token = prompt_tokens[0];
    int pos = 0;
    int next;
    // float nll = 0.0f;
    float prob = 0.0f;
    float cprob = 0.0f;
    float log_likelihood = 0.0f;
    float l = 0.0f;
    steps = num_prompt_tokens;
    while(pos < steps){

        float* logits = forward(transformer, token, pos);

        next = sample(sampler, logits);
        softmax(logits, sampler->vocab_size);
        prob = logits[next];
        // printf("next = %d, max_prob=%f, ll = %f\n", next, prob, log(prob));
        // printf("step %d - prob = %f\n", pos, prob);

        log_likelihood += log(prob);
        cprob += prob;
        l += log2(prob);

        // advance the state state machine
        // if (pos < num_prompt_tokens - 1) {
        //     // if we are still processing the input prompt, force the next prompt token
        //     next = prompt_tokens[pos + 1];
        // } else {
        //     // otherwise sample the next token from the logits
        //     next = sample(sampler, logits);
        //     softmax(logits, sampler->vocab_size);
        //     printf("next = %d, max_logit=%f\n", next, logits[next]);
        // }
        //printf("%d/%d\n", pos, steps);
        pos++;

        // token = prompt_tokens[pos];
        token = next;
    }
    nll = -log_likelihood;
    nll /= steps;
    cprob /= steps;
    l /= steps;
    // printf("nll = %f\n", nll/steps);
    printf("cprob = %f, pll?= %f, ppl2? = %f ppl3 = %f\n", cprob, pow(cprob, -1/steps), exp(nll), pow(2, -l));
    return l;
}

void calcualate_nll_for_file(Tokenizer* tokenizer, Transformer* transformer, Sampler *sampler, char* perplexity_input_file, int chunk_size, int steps){
    // read pp file in a buffer
    FILE *file;
    char *pp_text_buffer;
    long file_size;

    file = fopen(perplexity_input_file, "r");
    if (file == NULL){
        perror("ERROR opening file");
        exit(EXIT_FAILURE); 
    }
    // Get the size of the file
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);

    // Allocate memory for the buffer
    pp_text_buffer = (char *)malloc(chunk_size + 1);  // +1 for null-terminator
    // pp_text_buffer = (char *)malloc(file_size + 1);  // +1 for null-terminator
    if (pp_text_buffer == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // pp_text_buffer[file_size] = '\0';
    int num_chunks      = file_size / chunk_size;
    int last_chunk_size = file_size % chunk_size;

    int* prompt_tokens = (int*)malloc((chunk_size+3) * sizeof(int));
    if (prompt_tokens == NULL) {
        perror("Memory allocation for token array failed");
        exit(EXIT_FAILURE); 
    }
    // printf("DEBUG: num of chunks = %d, last_chunk_size = %d, total file size = %ld\n", num_chunks, last_chunk_size, file_size);
    //exit(EXIT_SUCCESS);
    float nll = 0.0f;

    for (int chunk = 0; chunk < num_chunks; chunk++){
        if (chunk == num_chunks-1) {
            printf("processing last chunk...\n");
            chunk_size = last_chunk_size;
        }
        // Read the chunk
        size_t read_size = fread(pp_text_buffer, 1, chunk_size, file);
        if (read_size != chunk_size) {
            perror("Error reading file");
            fclose(file);
            free(pp_text_buffer);
            exit(EXIT_FAILURE);
        }
            // Null-terminate the buffer
            pp_text_buffer[chunk_size] = '\0';

            // Print the contents of the buffer
            // puts("-----------------------------------");
            // printf("Buffer content:\n%s", pp_text_buffer);
            // puts("-----------------------------------");
            
            assert_eq(chunk_size, strlen(pp_text_buffer));
            // printf("size of pp text buffer to load at a time = %d, strlen = %lu\n", chunk_size, strlen(pp_text_buffer));
            

            int num_prompt_tokens = 0; // the total number of prompt tokens
            encode(tokenizer, pp_text_buffer, chunk == 0 ? 1 : 0, 0, prompt_tokens, &num_prompt_tokens);
            // for(int i=0;i < 10; i++) printf("prompt_tokens[%d]=%d\n", i, prompt_tokens[i] );
            printf("\nchunk [%d/%d], num tokens [%d] first token = %d, last token = %d\n", chunk+1, num_chunks, num_prompt_tokens, prompt_tokens[0], prompt_tokens[num_prompt_tokens-1] );
            puts("---------------------------------------------------------");
            
            nll = calculate_nll_for_chunk(transformer, tokenizer, sampler, prompt_tokens, num_prompt_tokens, steps, nll);
            printf("nll = %f\n", nll);

    }
    // Close the file
    free(prompt_tokens);
    fclose(file);
    free(pp_text_buffer);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -f <string> path to a file fed to the perplexity code\n");

    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
        // let's verify that the Tokenizer works as expected

    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    char *perplexity_input_file = "tinystories-valid.txt.1000";
    float temperature = 0.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = "";        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    int chunk_size = 512;
    int vocab_size = 32000;

     // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'f') { perplexity_input_file = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    calcualate_nll_for_file(&tokenizer,&transformer, &sampler, perplexity_input_file, chunk_size, steps);

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
