#include <gtest/gtest.h>

#include "../src/probabilistic_suffix_tree.hpp"

#include <array>
#include <tuple>
#include <vector>

#include <seqan3/alphabet/nucleotide/all.hpp>
#include <seqan3/alphabet/nucleotide/dna4.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/range/container/bitcompressed_vector.hpp>

class ProbabilisticSuffixTreeTest : public ::testing::Test {
protected:
  void SetUp() override {
    using seqan3::operator""_dna4;
    sequence = seqan3::bitcompressed_vector<seqan3::dna4>{"GATTATA"_dna4};
    probabilisticSuffixTree =
        pst::ProbabilisticSuffixTree{"TEST", sequence, 3, 2, 0.0};
  }

  seqan3::bitcompressed_vector<seqan3::dna4> sequence;
  pst::ProbabilisticSuffixTree<seqan3::dna4> probabilisticSuffixTree;
};

TEST_F(ProbabilisticSuffixTreeTest, ConstructorTable) {
  std::vector<int> expected_table{
      0, 2,  // root
      1, 10, // A
      0, 0,  // GATTATA-
      2, 14, // T
      7, 0,  // -
      2, 18, // AT
      7, 0,  // A-
      4, 22, // TA
      3, 0,  // TTATA-
      6, 0,  // ATA-
      3, 0,  // ATTATA-
      5, 0,  // TATA-
      7, 0   // TA-
  };

  EXPECT_EQ(probabilisticSuffixTree.table, expected_table);
}

TEST_F(ProbabilisticSuffixTreeTest, ConstructorStatus) {
  std::vector<pst::Status> expected_status{
      pst::Status::Included, // root
      pst::Status::Included, // A
      pst::Status::Excluded, // GATTATA-
      pst::Status::Included, // T
      pst::Status::Excluded, // -
      pst::Status::Included, // AT
      pst::Status::Excluded, // A-
      pst::Status::Included, // TA
      pst::Status::Excluded, // TTATA-
      pst::Status::Excluded, // ATA-
      pst::Status::Excluded, // ATTATA-
      pst::Status::Excluded, // TATA-
      pst::Status::Excluded  // TA-
  };

  EXPECT_EQ(probabilisticSuffixTree.status, expected_status);
}

TEST_F(ProbabilisticSuffixTreeTest, ConstructorSuffixLinks) {
  std::vector<int> expected_suffix_links{
      -1, // root
      0,  // A
      20, // GATTATA-
      0,  // T
      0,  // -
      6,  // AT
      8,  // A-
      2,  // TA
      22, // TTATA-
      24, // ATA-
      16, // ATTATA-
      18, // TATA-
      12  // TA-
  };

  EXPECT_EQ(probabilisticSuffixTree.suffix_links, expected_suffix_links);
}

TEST_F(ProbabilisticSuffixTreeTest, ConstructorProbabilities) {
  std::vector<std::array<float, seqan3::alphabet_size<seqan3::dna4>>>
      expected_probabilities{
          {4.0 / 11.0, 1.0 / 11.0, 2.0 / 11.0, 4.0 / 11.0}, // root
          {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 3.0 / 6.0},     // A
          {0, 0, 0, 0},                                 // GATTATA- (excluded)
          {3.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0}, // T
          {0, 0, 0, 0},                                 // - (excluded)
          {2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0}, // AT
          {0, 0, 0, 0},                                 // A- (excluded)
          {1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 2.0 / 5.0}, // TA
          {0, 0, 0, 0},                                 // TTATA- (excluded)
          {0, 0, 0, 0},                                 // ATA- (excluded)
          {0, 0, 0, 0},                                 // ATTATA- (excluded)
          {0, 0, 0, 0},                                 // TATA- (excluded)
          {0, 0, 0, 0}                                  // TA- (excluded)
      };

  for (int i = 0; i < probabilisticSuffixTree.probabilities.size() &&
                  i < expected_probabilities.size();
       i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_FLOAT_EQ(probabilisticSuffixTree.probabilities[i][j],
                      expected_probabilities[i][j]);
    }
  }
}

TEST_F(ProbabilisticSuffixTreeTest, PrunedKL) {
  probabilisticSuffixTree =
      pst::ProbabilisticSuffixTree{"TEST", sequence, 3, 2, 0.3};
  std::vector<pst::Status> expected_status{
      pst::Status::Included, // root
      pst::Status::Included, // A
      pst::Status::Excluded, // GATTATA-
      pst::Status::Excluded, // T
      pst::Status::Excluded, // -
      pst::Status::Excluded, // AT
      pst::Status::Excluded, // A-
      pst::Status::Excluded, // TA
      pst::Status::Excluded, // TTATA-
      pst::Status::Excluded, // ATA-
      pst::Status::Excluded, // ATTATA-
      pst::Status::Excluded, // TATA-
      pst::Status::Excluded  // TA-
  };

  EXPECT_EQ(probabilisticSuffixTree.status, expected_status);
}

TEST_F(ProbabilisticSuffixTreeTest, PrunedPS) {
  probabilisticSuffixTree =
      pst::ProbabilisticSuffixTree{"TEST", sequence, 3, 2};
  std::vector<pst::Status> expected_status{
      pst::Status::Included, // root
      pst::Status::Excluded, // A
      pst::Status::Excluded, // GATTATA-
      pst::Status::Excluded, // T
      pst::Status::Excluded, // -
      pst::Status::Excluded, // AT
      pst::Status::Excluded, // A-
      pst::Status::Excluded, // TA
      pst::Status::Excluded, // TTATA-
      pst::Status::Excluded, // ATA-
      pst::Status::Excluded, // ATTATA-
      pst::Status::Excluded, // TATA-
      pst::Status::Excluded  // TA-
  };

  EXPECT_EQ(probabilisticSuffixTree.status, expected_status);
}

TEST_F(ProbabilisticSuffixTreeTest, PrunedParameters) {
  probabilisticSuffixTree = pst::ProbabilisticSuffixTree{
      "TEST", sequence, 3, 2, 0.0, 6, "parameters", "KL"};
  std::vector<pst::Status> expected_status{
      pst::Status::Included, // root
      pst::Status::Included, // A
      pst::Status::Excluded, // GATTATA-
      pst::Status::Included, // T
      pst::Status::Excluded, // -
      pst::Status::Excluded, // AT
      pst::Status::Excluded, // A-
      pst::Status::Excluded, // TA
      pst::Status::Excluded, // TTATA-
      pst::Status::Excluded, // ATA-
      pst::Status::Excluded, // ATTATA-
      pst::Status::Excluded, // TATA-
      pst::Status::Excluded  // TA-
  };

  EXPECT_EQ(probabilisticSuffixTree.status, expected_status);
}

TEST_F(ProbabilisticSuffixTreeTest, Print) {
  pst::ProbabilisticSuffixTree<seqan3::dna4> pst_unpruned =
      pst::ProbabilisticSuffixTree{"TEST", sequence, 10, 0, 0.0};
  pst_unpruned.print();
  seqan3::debug_stream << std::endl;

  probabilisticSuffixTree.print();
  seqan3::debug_stream << std::endl;

  pst::ProbabilisticSuffixTree<seqan3::dna4> pst_pruned =
      pst::ProbabilisticSuffixTree{"TEST", sequence, 3, 2, 1.2};
  pst_pruned.print();
  seqan3::debug_stream << std::endl;
}

TEST(ProbabilisticSuffixTreeTestPS, PSPruning) {
  using seqan3::operator""_dna4;
  seqan3::bitcompressed_vector<seqan3::dna4> long_sequence{
      "AATAATAATAATAATAATAATAATCGCGCGCGCGCGCATATATAT"_dna4};
  pst::ProbabilisticSuffixTree<seqan3::dna4> pst_ps =
      pst::ProbabilisticSuffixTree{"TEST", long_sequence, 3, 2};
  pst_ps.print();
  seqan3::debug_stream << std::endl;
}
