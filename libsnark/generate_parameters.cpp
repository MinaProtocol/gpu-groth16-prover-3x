#include <cassert>
#include <cstdio>
#include <fstream>

#include <libff/common/rng.hpp>
#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>
#include <libsnark/serialization.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <omp.h>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libsnark/knowledge_commitment/kc_multiexp.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>
#include <libsnark/relations/constraint_satisfaction_problems/r1cs/examples/r1cs_examples.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>

using namespace libsnark;
using namespace libff;

const bool debug = false;

template<typename ppT>
int generate_paramaters(
    int log2_d,
    char* params_path, char* input_path)
{
    srand(time(NULL));
    setbuf(stdout, NULL);

    ppT::init_public_params();

    const size_t primary_input_size = 1;

    size_t d_plus_1 = 1 << log2_d;
    size_t d = d_plus_1 - 1;

    r1cs_example<Fr<ppT>> example = generate_r1cs_example_with_field_input<Fr<ppT>>(d-1, 1);
    r1cs_gg_ppzksnark_keypair<ppT> keypair = r1cs_gg_ppzksnark_generator<ppT>(example.constraint_system);

    r1cs_variable_assignment<Fr<ppT>> full_variable_assignment = example.primary_input;
    full_variable_assignment.insert(full_variable_assignment.end(), example.auxiliary_input.begin(), example.auxiliary_input.end());

    std::vector<Fr<ppT>> ca(d_plus_1, Fr<ppT>::zero()), cb(d_plus_1, Fr<ppT>::zero()), cc(d_plus_1, Fr<ppT>::zero());
    for (size_t i = 0; i <= primary_input_size; ++i)
    {
        ca[i+keypair.pk.constraint_system.num_constraints()] = (i > 0 ? full_variable_assignment[i-1] : Fr<ppT>::one());
    }
    for (size_t i = 0; i < keypair.pk.constraint_system.num_constraints(); ++i)
    {
        ca[i] += keypair.pk.constraint_system.constraints[i].a.evaluate(full_variable_assignment);
        cb[i] += keypair.pk.constraint_system.constraints[i].b.evaluate(full_variable_assignment);
    }
    for (size_t i = 0; i < keypair.pk.constraint_system.num_constraints(); ++i)
    {
        cc[i] += keypair.pk.constraint_system.constraints[i].c.evaluate(full_variable_assignment);
    }

    // Write parameters
    auto params = fopen(params_path, "w");
    size_t m = example.constraint_system.num_variables();

    write_size_t(params, d);
    write_size_t(params, m);

    for (size_t i = 0; i <= m; ++i) {
      write_g1<ppT>(params, keypair.pk.A_query[i]);
    }

    for (size_t i = 0; i <= m; ++i) {
      write_g1<ppT>(params, keypair.pk.B_query[i].h);
    }

    for (size_t i = 0; i <= m; ++i) {
      write_g2<ppT>(params, keypair.pk.B_query[i].g);
    }

    for (size_t i = 0; i < m-1; ++i) {
      write_g1<ppT>(params, keypair.pk.L_query[i]);
    }

    for (size_t i = 0; i < d; ++i) {
      write_g1<ppT>(params, keypair.pk.H_query[i]);
    }
    fclose(params);

    // Write input
    auto input = fopen(input_path, "w");

    write_fr<ppT>(input, Fr<ppT>::one());
    for (size_t i = 0; i < m; ++i) {
      write_fr<ppT>(input, full_variable_assignment[i]);
    }

    for (size_t i = 0; i < d_plus_1; ++i) {
      write_fr<ppT>(input, ca[i]);
    }
    for (size_t i = 0; i < d_plus_1; ++i) {
      write_fr<ppT>(input, cb[i]);
    }
    for (size_t i = 0; i < d_plus_1; ++i) {
      write_fr<ppT>(input, cc[i]);
    }

    const libff::Fr<ppT> r = libff::Fr<ppT>::random_element();
    write_fr<ppT>(input, r);

    fclose(input);

    if (debug) {
      std::ofstream vk_debug;
      vk_debug.open("verification-key.debug");
      vk_debug << keypair.vk;
      vk_debug.close();

      std::ofstream pk_debug;
      pk_debug.open("proving-key.debug");
      pk_debug << keypair.pk;
      pk_debug.close();
    }

    return 0;
}

int main(int argc, const char * argv[])
{
  int log2_d_4753 = 20, log2_d_6753 = 15;
  if (argc > 1) {
    std::string fastflag(argv[1]);
    if (fastflag == "fast") {
      log2_d_4753 = 14;
      log2_d_6753 = 10;
    }
  }
  generate_paramaters<mnt4753_pp>(log2_d_4753, "MNT4753-parameters", "MNT4753-input");
  generate_paramaters<mnt6753_pp>(log2_d_6753, "MNT6753-parameters", "MNT6753-input");
}
