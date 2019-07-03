#include <cassert>
#include <cstdio>
#include <fstream>
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
#include <libsnark/knowledge_commitment/knowledge_commitment.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>

#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>

#include <libfqfft/evaluation_domain/domains/basic_radix2_domain.hpp>

using namespace libff;
using namespace libsnark;

const multi_exp_method method = multi_exp_method_BDLO12;
// const multi_exp_method method = multi_exp_method_bos_coster;

template<typename ppT>
class groth16_parameters {
  public:
    size_t d;
    size_t m;
    std::vector<G1<ppT>> A, B1, L, H;
    std::vector<G2<ppT>> B2;

  groth16_parameters(const char* path) {
    FILE* params = fopen(path, "r");
    d = read_size_t(params);
    m = read_size_t(params);
    for (size_t i = 0; i <= m; ++i) { A.emplace_back(read_g1<ppT>(params)); }
    for (size_t i = 0; i <= m; ++i) { B1.emplace_back(read_g1<ppT>(params)); }
    for (size_t i = 0; i <= m; ++i) { B2.emplace_back(read_g2<ppT>(params)); }
    for (size_t i = 0; i < m-1; ++i) { L.emplace_back(read_g1<ppT>(params)); }
    for (size_t i = 0; i < d; ++i) { H.emplace_back(read_g1<ppT>(params)); }
    fclose(params);
  }
};

template<typename ppT>
class groth16_input {
  public:
    std::vector<Fr<ppT>> w;
    std::vector<Fr<ppT>> ca, cb, cc;
    Fr<ppT> r;

  groth16_input(const char* path, size_t d, size_t m) {
    FILE* inputs = fopen(path, "r");

    for (size_t i = 0; i < m + 1; ++i) { w.emplace_back(read_fr<ppT>(inputs)); }

    for (size_t i = 0; i < d + 1; ++i) { ca.emplace_back(read_fr<ppT>(inputs)); }
    for (size_t i = 0; i < d + 1; ++i) { cb.emplace_back(read_fr<ppT>(inputs)); }
    for (size_t i = 0; i < d + 1; ++i) { cc.emplace_back(read_fr<ppT>(inputs)); }

    r = read_fr<ppT>(inputs);

    fclose(inputs);
  }
};

template<typename ppT>
class groth16_output {
  public:
    G1<ppT> A, C;
    G2<ppT> B;

  groth16_output(G1<ppT> &&A, G2<ppT> &&B, G1<ppT> &&C) :
    A(std::move(A)), B(std::move(B)), C(std::move(C)) {}

  void write(const char* path) {
    FILE* out = fopen(path, "w");
    write_g1<ppT>(out, A);
    write_g2<ppT>(out, B);
    write_g1<ppT>(out, C);
    fclose(out);
  }
};

// Here is where all the FFTs happen.
template<typename ppT>
std::vector<Fr<ppT>> compute_H(
    size_t d,
    std::vector<Fr<ppT>> &ca,
    std::vector<Fr<ppT>> &cb,
    std::vector<Fr<ppT>> &cc)
{
    // Begin witness map
    libff::enter_block("Compute the polynomial H");

    const std::shared_ptr<libfqfft::evaluation_domain<Fr<ppT>> > domain = libfqfft::get_evaluation_domain<Fr<ppT>>(d + 1);

    domain->iFFT(ca);
    domain->iFFT(cb);

    domain->cosetFFT(ca, Fr<ppT>::multiplicative_generator);
    domain->cosetFFT(cb, Fr<ppT>::multiplicative_generator);

    libff::enter_block("Compute evaluation of polynomial H on set T");
    std::vector<Fr<ppT>> &H_tmp = ca; // can overwrite ca because it is not used later
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        H_tmp[i] = ca[i]*cb[i];
    }
    std::vector<Fr<ppT>>().swap(cb); // destroy cb

    domain->iFFT(cc);

    domain->cosetFFT(cc, Fr<ppT>::multiplicative_generator);

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        H_tmp[i] = (H_tmp[i]-cc[i]);
    }

    domain->divide_by_Z_on_coset(H_tmp);

    libff::leave_block("Compute evaluation of polynomial H on set T");

    domain->icosetFFT(H_tmp, Fr<ppT>::multiplicative_generator);

    std::vector<Fr<ppT>> coefficients_for_H(domain->m+1, Fr<ppT>::zero());
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        coefficients_for_H[i] = H_tmp[i];
    }

    libff::leave_block("Compute the polynomial H");

    return coefficients_for_H;
}

template<typename G, typename Fr>
G multiexp(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length)
{
#ifdef MULTICORE
    const size_t chunks = omp_get_max_threads(); // to override, set OMP_NUM_THREADS env var or call omp_set_num_threads()
#else
    const size_t chunks = 1;
#endif

    return libff::multi_exp_with_mixed_addition<G,
                                                Fr,
                                                method>(
        g_start,
        g_start + length,
        scalar_start,
        scalar_start + length,
        chunks);

}

template<typename ppT>
int run_prover(
    const char* params_path,
    const char* input_path,
    const char* output_path)
{
    ppT::init_public_params();

    const size_t primary_input_size = 1;

    const groth16_parameters<ppT> parameters(params_path);
    const groth16_input<ppT> input(input_path, parameters.d, parameters.m);

    std::vector<Fr<ppT>> w  = std::move(input.w);
    std::vector<Fr<ppT>> ca = std::move(input.ca);
    std::vector<Fr<ppT>> cb = std::move(input.cb);
    std::vector<Fr<ppT>> cc = std::move(input.cc);

    // End reading of parameters and input

    libff::enter_block("Call to r1cs_gg_ppzksnark_prover");

    std::vector<Fr<ppT>> coefficients_for_H = compute_H<ppT>(
        parameters.d,
        ca, cb, cc);

    libff::enter_block("Compute the proof");
    libff::enter_block("Multi-exponentiations");

    // Now the 5 multi-exponentiations
    G1<ppT> evaluation_At = multiexp<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.A.begin(), parameters.m + 1);

    G1<ppT> evaluation_Bt1 = multiexp<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.B1.begin(), parameters.m + 1);

    G2<ppT> evaluation_Bt2 = multiexp<G2<ppT>, Fr<ppT>>(
        w.begin(), parameters.B2.begin(), parameters.m + 1);

    G1<ppT> evaluation_Ht = multiexp<G1<ppT>, Fr<ppT>>(
        coefficients_for_H.begin(), parameters.H.begin(), parameters.d);

    G1<ppT> evaluation_Lt = multiexp<G1<ppT>, Fr<ppT>>(
        w.begin() + primary_input_size + 1,
        parameters.L.begin(),
        parameters.m - 1);

    libff::G1<ppT> C = evaluation_Ht + evaluation_Lt + input.r * evaluation_Bt1; /*+ s *  g1_A  - (r * s) * pk.delta_g1; */

    libff::leave_block("Multi-exponentiations");
    libff::leave_block("Compute the proof");
    libff::leave_block("Call to r1cs_gg_ppzksnark_prover");

    groth16_output<ppT> output(
      std::move(evaluation_At),
      std::move(evaluation_Bt2),
      std::move(C));

    output.write(output_path);

    return 0;
}

template<typename ppT>
void write_g1_vec(FILE *out, const std::vector<G1<ppT>> &vec) {
    for (const auto &P : vec)
        write_g1<ppT>(out, P);
}

template<typename ppT>
void write_g2_vec(FILE *out, const std::vector<G2<ppT>> &vec) {
    for (const auto &P : vec)
        write_g2<ppT>(out, P);
}

template<typename ppT>
void output_g1_multiples(int C, const std::vector<G1<ppT>> &vec, FILE *output) {
    // If vec = [P0, ..., Pn], then multiples holds an array
    //
    // [    P0, ...,     Pn,
    //     2P0, ...,    2Pn,
    //     3P0, ...,    3Pn,
    //          ...,
    //  2^(C-1) P0, ..., 2^(C-1) Pn]
    std::vector<G1<ppT>> multiples;
    size_t len = vec.size();
    multiples.resize(len * ((1U << C) - 1));
    std::copy(vec.begin(), vec.end(), multiples.begin());

    for (size_t i = 1; i < (1U << C) - 1; ++i) {
        size_t prev_row_offset = (i-1)*len;
        size_t curr_row_offset = i*len;
#ifdef MULTICORE
#pragma omp parallel for
#endif
        for (size_t j = 0; j < len; ++j)
           multiples[curr_row_offset + j] = vec[j] + multiples[prev_row_offset + j];
    }

    if (multiples.size() != ((1U << C) - 1)*len) {
        fprintf(stderr, "Broken preprocessing table: got %zu, expected %zu\n",
                multiples.size(), ((1U << C) - 1) * len);
        abort();
    }
    write_g1_vec<ppT>(output, multiples);
}

template<typename ppT>
void output_g2_multiples(int C, const std::vector<G2<ppT>> &vec, FILE *output) {
    // If vec = [P0, ..., Pn], then multiples holds an array
    //
    // [    P0, ...,     Pn,
    //     2P0, ...,    2Pn,
    //     3P0, ...,    3Pn,
    //          ...,
    //  2^(C-1) P0, ..., 2^(C-1) Pn]
    std::vector<G2<ppT>> multiples;
    size_t len = vec.size();
    multiples.resize(len * ((1U << C) - 1));
    std::copy(vec.begin(), vec.end(), multiples.begin());

    for (size_t i = 1; i < (1U << C) - 1; ++i) {
        size_t prev_row_offset = (i-1)*len;
        size_t curr_row_offset = i*len;
#ifdef MULTICORE
#pragma omp parallel for
#endif
        for (size_t j = 0; j < len; ++j)
           multiples[curr_row_offset + j] = vec[j] + multiples[prev_row_offset + j];
    }

    if (multiples.size() != ((1U << C) - 1)*len) {
        fprintf(stderr, "Broken preprocessing table: got %zu, expected %zu\n",
                multiples.size(), ((1U << C) - 1) * len);
        abort();
    }
    write_g2_vec<ppT>(output, multiples);
}

template <typename ppT>
void run_preprocess(const char *params_path, const char *output_path)
{
    ppT::init_public_params();

    const groth16_parameters<ppT> params(params_path);

    // We will produce 2^C precomputed points [i]P for i = 1..2^C
    // for every input point P
    static constexpr size_t C = 5;

    size_t d = params.d, m =  params.m;
    printf("d = %zu, m = %zu, C = %zu\n", d, m, C);

    FILE *output = fopen(output_path, "w");

//    printf("Processing A...\n");
//    output_g1_multiples<ppT>(C, params.A, output);
    printf("Processing B1...\n");
    output_g1_multiples<ppT>(C, params.B1, output);
    printf("Processing B2...\n");
    output_g2_multiples<ppT>(C, params.B2, output);
    printf("Processing L...\n");
    output_g1_multiples<ppT>(C, params.L, output);
//    printf("Processing H...\n");
//    output_g1_multiples<ppT>(C, params.H, output);

    fclose(output);
}


int main(int argc, const char * argv[])
{
  setbuf(stdout, NULL);
  std::string curve(argv[1]);
  std::string mode(argv[2]);

  const char* params_path = argv[3];

  if (mode == "compute") {
      const char *input_path = argv[4];
      const char *output_path = argv[5];

      if (curve == "MNT4753") {
          run_prover<mnt4753_pp>(params_path, input_path, output_path);
      } else if (curve == "MNT6753") {
          run_prover<mnt6753_pp>(params_path, input_path, output_path);
      }
  } else if (mode == "preprocess") {
      if (curve == "MNT4753") {
          run_preprocess<mnt4753_pp>(params_path, "MNT4753_preprocessed");
      } else if (curve == "MNT6753") {
          run_preprocess<mnt6753_pp>(params_path, "MNT6753_preprocessed");
      }
  }
}

template<typename ppT>
void debug(
    Fr<ppT>& r,
    groth16_output<ppT>& output,
    std::vector<Fr<ppT>>& w) {

    const size_t primary_input_size = 1;

    std::vector<Fr<ppT>> primary_input(w.begin() + 1, w.begin() + 1 + primary_input_size);
    std::vector<Fr<ppT>> auxiliary_input(w.begin() + 1 + primary_input_size, w.end() );

    const libff::Fr<ppT> s = libff::Fr<ppT>::random_element();

    r1cs_gg_ppzksnark_proving_key<ppT> pk;
    std::ifstream pk_debug;
    pk_debug.open("proving-key.debug");
    pk_debug >> pk;

    /* A = alpha + sum_i(a_i*A_i(t)) + r*delta */
    libff::G1<ppT> g1_A = pk.alpha_g1 + output.A + r * pk.delta_g1;

    /* B = beta + sum_i(a_i*B_i(t)) + s*delta */
    libff::G2<ppT> g2_B = pk.beta_g2 + output.B + s * pk.delta_g2;

    /* C = sum_i(a_i*((beta*A_i(t) + alpha*B_i(t) + C_i(t)) + H(t)*Z(t))/delta) + A*s + r*b - r*s*delta */
    libff::G1<ppT> g1_C = output.C + s * g1_A + r * pk.beta_g1;

    libff::leave_block("Compute the proof");

    libff::leave_block("Call to r1cs_gg_ppzksnark_prover");

    r1cs_gg_ppzksnark_proof<ppT> proof = r1cs_gg_ppzksnark_proof<ppT>(std::move(g1_A), std::move(g2_B), std::move(g1_C));
    proof.print_size();

    r1cs_gg_ppzksnark_verification_key<ppT> vk;
    std::ifstream vk_debug;
    vk_debug.open("verification-key.debug");
    vk_debug >> vk;
    vk_debug.close();

    assert (r1cs_gg_ppzksnark_verifier_strong_IC<ppT>(vk, primary_input, proof) );

    r1cs_gg_ppzksnark_proof<ppT> proof1=
      r1cs_gg_ppzksnark_prover<ppT>(
          pk, 
          primary_input,
          auxiliary_input);
    assert (r1cs_gg_ppzksnark_verifier_strong_IC<ppT>(vk, primary_input, proof1) );
}
