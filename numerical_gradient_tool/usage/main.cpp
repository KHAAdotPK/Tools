/*
    numerical_gradient/usage/main.cpp
    Writteb by, Q@khaa.pk
 */

#include "main.hh"

template <typename E = double>
E calculate_loss(Collective<E>& W1_epsilon, Collective<E>& W2, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{
    E loss = 0;
    cc_tokenizer::string_character_traits<char>::size_type number_of_context_words = SKIP_GRAM_CONTEXT_WINDOW_SIZE*2;

    try
    {
        // Lookup embedding vector for the target word
        Collective<E> W1_epsilon_slice =  W1_epsilon.slice((pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)* W1_epsilon.getShape().getNumberOfColumns(), W1_epsilon.getShape().getNumberOfColumns());

        // Compute logits (unnormalized scores) for all vocabulary words
        Collective<E> logits = Numcy::dot(W1_epsilon_slice, W2); 

        // Apply softmax to convert logits into probabilities
        Collective<E> exp_logits = Numcy::exp<E>(logits - Numcy::max<E>(logits));
        Collective<E> probabilities = exp_logits / Numcy::sum(exp_logits);
        
        // Compute loss
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_CONTEXT_WINDOW_SIZE; i++)
        {
            if ((*(pair->getLeft()))[i] != INDEX_NOT_FOUND_AT_VALUE)
            {
                loss = loss + ((-1) * std::log(probabilities[(*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE]));
            }
            else
            {
                number_of_context_words = number_of_context_words - 1;
            }
        }
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_CONTEXT_WINDOW_SIZE; i++)
        {
            if ((*(pair->getRight()))[i] != INDEX_NOT_FOUND_AT_VALUE)
            {
                loss = loss + ((-1) * std::log(probabilities[(*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE]));
            }
            else
            {
                number_of_context_words = number_of_context_words - 1;
            }
        }        
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("Calculate_loss() -> ") + e.what());
    }

    return (loss/number_of_context_words /*(SKIP_GRAM_CONTEXT_WINDOW_SIZE*2)*/);        
}

template <typename E = double>
void numerical_gradient(Collective<E>& W1, Collective<E>& W2, Collective<E>& NG, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{   
    E loss_plus = 0, loss_minus = 0; 

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
    {
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W1.getShape().getNumberOfColumns(); j++)
        {
            E original_value = W1[i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + j];

            // Perturb positively
            W1[i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + j] = original_value + EPSILON_SMALL_PERTURBATION; 
            try
            {
                loss_plus = calculate_loss<E>(W1, W2, pair);
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("numerical_gradient() -> ") + e.what()); 
            }

            // Perturb negatively
            W1[i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + j] = original_value - EPSILON_SMALL_PERTURBATION; 
            try
            {
                loss_minus = calculate_loss<E>(W1, W2, pair);
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("numerical_gradient() -> ") + e.what()); 
            }
                        
            NG[i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + j] = (loss_plus - loss_minus) / (2 * EPSILON_SMALL_PERTURBATION); 
                        
            W1[i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE + j] = original_value;            
            loss_plus = 0;
            loss_minus = 0;
        }
    }    
}

int main(int argc, char* argv[])
{
    ARG arg_corpus, arg_help, arg_input, arg_verbose, arg_ng;
    cc_tokenizer::String<char> data;
    CORPUS corpora;
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));

    if (argc < 2)
    {        
        HELP(argsv_parser, arg_help, "help");                
        HELP_DUMP(argsv_parser, arg_help);

        return 0;                     
    }

    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    FIND_ARG(argv, argc, argsv_parser, "corpus", arg_corpus);
    FIND_ARG(argv, argc, argsv_parser, "--input", arg_input);
    FIND_ARG(argv, argc, argsv_parser, "verbose", arg_verbose);
    FIND_ARG(argv, argc, argsv_parser, "--output", arg_ng);

    if (arg_ng.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_ng); 
        if (!arg_ng.argc)
        {
            ARG arg_ng_help;
            HELP(argsv_parser, arg_ng_help, "output");                
            HELP_DUMP(argsv_parser, arg_ng_help);

            return 0;
        }
    }

    if (arg_corpus.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_corpus);
        if (arg_corpus.argc)
        {            
            try 
            {
                data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + 1]);

                if (arg_verbose.i)
                {
                    std::cout<< "Corpus: " << argv[arg_corpus.i + 1] << std::endl;
                }

                corpora = CORPUS(data);
            }
            catch (ala_exception e)
            {                
                cc_tokenizer::String<char> message = cc_tokenizer::String<char>("main() -> ") +  e.what();
                std::cerr << message.c_str() << std::endl;

                return -1;
            }            
        }
        else
        { 
            ARG arg_corpus_help;
            HELP(argsv_parser, arg_corpus_help, "--corpus");                
            HELP_DUMP(argsv_parser, arg_corpus_help);

            return 0; 
        }                               
    }

    Collective<double> W1, W2, NG/* Numerical gradients */ ;

    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_input); 
    if (arg_input.argc > 1)
    {
        W1 = Collective<double>{NULL, DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, corpora.numberOfUniqueTokens(), NULL, NULL}};
        W2 = Collective<double>{NULL, DIMENSIONS{corpora.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
        
        try
        {                    
            READ_W_BIN(W1, cc_tokenizer::String<char>(argv[arg_input.i + 1]), double);
            READ_W_BIN(W2, cc_tokenizer::String<char>(argv[arg_input.i + 2]), double);

            NG = Numcy::zeros<double>(*(W1.getShape().copy()));            
        }
        catch (ala_exception& e)
        {
            std::cerr<< "main() -> " << e.what() << std::endl;

            return 0;
        }

        if (arg_verbose.i)
        {
            std::cout<< "Dimensions W1 = " << W1.getShape().getNumberOfColumns() <<"x" << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            std::cout<< "Dimensions W2 = " << W2.getShape().getNumberOfColumns() <<"x" << W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
        }
    }
    else
    {
        ARG arg_input_help;
        HELP(argsv_parser, arg_input_help, "--input");                
        HELP_DUMP(argsv_parser, arg_input_help);

        return 0;
    }

    Collective<double> W2_transpose;

    try 
    {
        W2_transpose = Numcy::transpose(W2);

        if (arg_verbose.i)
        {
            std::cout<< "Dimensions W2(transposed) = " << W2_transpose.getShape().getNumberOfColumns() <<"x" << W2_transpose.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
        }
    }
    catch (ala_exception& e)
    {
        std::cerr << "main() -> " << e.what() << std::endl;

        return 0;
    }

    PAIRS pairs(corpora, arg_verbose.i ? true : false);

    while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
    {
        try
        {
            numerical_gradient(W1, W2, NG, pairs.get_current_word_pair());            
        }
        catch(ala_exception& e)
        {
            std::cerr << "main() -> " << e.what() << std::endl;

            return 0;
        }        
    }

    if (arg_ng.i)
    {                
        WRITE_W_BIN(NG, cc_tokenizer::String<char>(argv[arg_ng.i + 1]), double);                
    }
    else
    {
        WRITE_W_BIN(NG, cc_tokenizer::String<char>(DEFAULT_NUMERICAL_GRADIENT_FILE_NAME), double);
    }

    if (arg_verbose.i)
    {
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < NG.getShape().getN(); i++)
        {
            std::cout<< NG[i] << ", ";
        }
        std::cout<< std::endl;
    }

    return 0;
}
