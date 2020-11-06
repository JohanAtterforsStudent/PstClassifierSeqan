#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <thread>

#include <highfive/H5File.hpp>

#include <Eigen/Dense>

#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/argument_parser/argument_parser.hpp>
#include <seqan3/io/sequence_file/input.hpp>

#include "distances/composition_vectors.hpp"
#include "distances/negative_log_likelihood.hpp"
#include "distances/parallelize.hpp"
#include "probabilistic_suffix_tree_map.hpp"

using tree_t = pst::ProbabilisticSuffixTreeMap<seqan3::dna5>;

struct input_arguments {
  int background_order{0};
  std::filesystem::path filepath{};
  std::filesystem::path outpath{};
  std::filesystem::path sequence_list{};
};

input_arguments parse_cli_arguments(int argc, char *argv[]) {
  input_arguments arguments{};

  seqan3::argument_parser parser{"score-sequences", argc, argv, false};
  parser.info.short_description = "Calculates the negative log-likelihood of a "
                                  "set of sequences for a set of signatures.";

  parser.add_option(arguments.filepath, 'p', "path",
                    "Path to hdf5 file where PSTs are stored.");
  parser.add_option(arguments.outpath, 'o', "out-path",
                    "Path to hdf5 file where scores will be stored.");
  parser.add_option(arguments.sequence_list, 's', "sequence-list",
                    "Path to a file with paths to sequences.");
  parser.add_option(arguments.background_order, 'b', "background-order",
                    "Length of background for log-likelihood.");

  try {
    parser.parse();
  } catch (seqan3::argument_parser_error const &ext) {
    std::cout << "[PARSER ERROR] " << ext.what() << '\n';
    return arguments;
  }

  return arguments;
}

std::vector<tree_t> get_trees(HighFive::File &file) {
  const std::string DATASET_NAME("signatures");

  HighFive::DataSet dataset = file.getDataSet(DATASET_NAME);

  std::vector<std::string> result_string_list;
  dataset.read(result_string_list);

  std::vector<tree_t> trees{};
  std::transform(result_string_list.begin(), result_string_list.end(),
                 std::back_inserter(trees),
                 [](std::string &tree) -> tree_t { return tree_t{tree}; });

  return trees;
}

void score_slice(
    int start_index, int stop_index, std::vector<double> &scores,
    std::vector<tree_t> &trees, std::vector<seqan3::dna5> &sequence,
    const std::function<float(tree_t &, std::vector<seqan3::dna5> &)> &fun) {

  for (int i = start_index; i < stop_index; i++) {
    scores[i] = fun(trees[i], sequence);
  }
}

std::vector<std::vector<double>>
score_sequences_paths(std::vector<tree_t> &trees,
                      std::vector<std::string> &sequence_list,
                      int background_order) {
  std::vector<std::vector<double>> scores{};

  int sequence_idx = 0;

  for (auto &path : sequence_list) {
    seqan3::sequence_file_input file_in{std::ifstream{path},
                                        seqan3::format_fasta{}};
    for (auto &[seq, id, qual] : file_in) {
      std::vector<double> scores_row(trees.size());

      auto fun = [&](int start_index, int stop_index) {
        score_slice(
            start_index, stop_index, std::ref(scores_row), std::ref(trees),
            std::ref(seq),
            pst::distances::negative_log_likelihood_symmetric<seqan3::dna5>);
      };

      pst::parallelize::parallelize(trees.size(), fun);

      scores.push_back(scores_row);
    }
  }

  return scores;
}

std::vector<std::vector<double>>
score_sequences(std::vector<tree_t> &trees, std::vector<std::string> &sequences,
                int background_order) {
  std::vector<std::vector<double>> scores{};

  int sequence_idx = 0;

  for (auto &sequence : sequences) {
    std::vector<seqan3::dna5> seq = sequence |
                                    seqan3::views::char_to<seqan3::dna5> |
                                    seqan3::views::to<std::vector>;

    std::vector<double> scores_row(trees.size());

    auto fun = [&](int start_index, int stop_index) {
      score_slice(
          start_index, stop_index, std::ref(scores_row), std::ref(trees),
          std::ref(seq),
          pst::distances::negative_log_likelihood_symmetric<seqan3::dna5>);
    };

    pst::parallelize::parallelize(trees.size(), fun);

    scores.push_back(scores_row);
  }

  return scores;
}

static std::vector<std::vector<double>>
score_cpp(std::vector<std::string> tree_strings,
          std::vector<std::string> sequences) {
  std::vector<tree_t> trees{};

  std::transform(tree_strings.begin(), tree_strings.end(),
                 std::back_inserter(trees),
                 [](std::string &tree) -> tree_t { return tree_t{tree}; });

  auto scores = score_sequences(trees, sequences, 0);

  return scores;
}

int main(int argc, char *argv[]) {
  input_arguments arguments = parse_cli_arguments(argc, argv);

  HighFive::File file{arguments.filepath, HighFive::File::ReadOnly};
  HighFive::File out_file{arguments.outpath, HighFive::File::ReadWrite |
                                                 HighFive::File::OpenOrCreate};

  std::ifstream infile(arguments.sequence_list);
  std::string path;
  std::vector<std::string> sequence_list{};
  while (std::getline(infile, path)) {
    sequence_list.push_back(path);
  }

  auto trees = get_trees(file);
  auto scores =
      score_sequences_paths(trees, sequence_list, arguments.background_order);

  if (!out_file.exist("scores")) {
    std::vector<size_t> dims{scores.size(),
                             static_cast<size_t>(scores[0].size())};
    out_file.createDataSet<double>("scores", HighFive::DataSpace(dims));
  }
  auto scores_dataset = out_file.getDataSet("scores");

  scores_dataset.write(scores);

  return EXIT_SUCCESS;
}