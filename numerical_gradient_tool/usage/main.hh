/*
    numerical_gradient_tool/usage/main.hh
    Written by, Q@khaa.pk
 */

#include <iostream>

#ifndef TRAINED_WEIGHTS_NUMERICAL_GRADIENT_TOOL_HH
#define TRAINED_WEIGHTS_NUMERICAL_GRADIENT_TOOL_HH

#define EPSILON_SMALL_PERTURBATION 1e-5
#define DEFAULT_NUMERICAL_GRADIENT_FILE_NAME "NG.dat"

#define COMMAND "h -h help --help ? /? (Displays the help screen, listing available commands and their descriptions.)\n\
v -v version --version /v (Shows the current version of the software.)\n\
corpus --corpus (Specifies the path to the file containing the training data.)\n\
verbose --verbose (Enables detailed output for each operation during execution.)\n\
input --input (Specifies the filenames to retrieve the input W1 and output W2 trained weights.)\n\
output --output (Specifies the filename to store the newly calculated numerical gradients.)\n"

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#define GRAMMAR_END_OF_TOKEN_MARKER ' '

#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif
#define GRAMMAR_END_OF_LINE_MARKER '\n'

#ifdef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE
#undef SKIP_GRAM_EMBEDDNG_VECTOR_SIZE
#endif
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 64

#ifdef SKIP_GRAM_REGULARIZATION_STRENGTH
#undef SKIP_GRAM_REGULARIZATION_STRENGTH
#endif
#define SKIP_GRAM_REGULARIZATION_STRENGTH 0.00

#ifdef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#undef SKIP_GRAM_CONTEXT_WINDOW_SIZE
#endif
#define SKIP_GRAM_CONTEXT_WINDOW_SIZE 2

#include "../lib/argsv-cpp/lib/parser/parser.hh"
#include "../lib/sundry/cooked_read_new.hh"
#include "../lib/sundry/cooked_write_new.hh"
#include "../lib/corpus/corpus.hh"
#include "../lib/Numcy/header.hh"
#include "../lib/read_write_weights/header.hh"
#include "../lib/pairs/src/header.hh"

#endif
