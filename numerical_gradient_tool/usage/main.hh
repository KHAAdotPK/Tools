/*
    numerical_gradient_tool/usage/main.hh
    Written by, Q@khaa.pk
 */

#include <iostream>

#ifndef TRAINED_WEIGHTS_NUMERICAL_GRADIENT_TOOL_HH
#define TRAINED_WEIGHTS_NUMERICAL_GRADIENT_TOOL_HH

/*
    Commonly set to 1𝑒−5 or 1𝑒−6, as these values balance accuracy and stability for most cases.
 */
#define EPSILON_SMALL_PERTURBATION_FOR_NUMERICAL_GRADIENT 1e-5
/*
    This 𝜖 is added to the denominator in the relative error formula to avoid division by zero when both gradient_W1 and NG(Numerical Gradient) are zero or very close to zero.
    A much smaller value, like 1𝑒−8, ensures stability without significantly impacting the relative error's accuracy.
 */
#define EPSILON_SMALL_PERTURBATION_FOR_RELATIVE_ERROR 1e-8

#define DEFAULT_NUMERICAL_GRADIENT_FILE_NAME "NG.dat"
#define DEFAULT_RELATIVE_ERROR_FILE_NAME "RE.dat"

#define COMMAND "h -h help --help ? /? (Displays the help screen, listing available commands and their descriptions.)\n\
v -v version --version /v (Shows the current version of the software.)\n\
corpus --corpus (Specifies the path to the file containing the training data.)\n\
verbose --verbose (Enables detailed output for each operation during execution.)\n\
input --input (Specifies the filenames to retrieve the input W1 and output W2 trained weights.)\n\
output --output (Specifies the filename to store the newly calculated numerical gradients.)\n\
re --re relative_error relativeError (To calculate the relative error between gradient_W1 and NG - Numerical Gradient. This option expects a file name to store the calculated relative errors.)\n"

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
