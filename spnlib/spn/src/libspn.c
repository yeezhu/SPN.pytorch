#include <TH/TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "libspn.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define spn_(NAME) TH_CONCAT_4(spn_, Real, _, NAME)

#include "generic/SoftProposalGenerator.c"
#include "THGenerateFloatTypes.h"