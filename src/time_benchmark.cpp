#include <functional>
#include <filesystem>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

#include <Eigen/Dense>

#include <highfive/H5File.hpp>

#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/argument_parser/argument_parser.hpp>

#include <seqan3/std/filesystem>

#include <indicators/cursor_control.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>

#include "pst/distances/cv.hpp"
#include "pst/distances/d2.hpp"
#include "pst/distances/d2star.hpp"
#include "pst/distances/dvstar.hpp"
#include "pst/distances/kl_divergence.hpp"
#include "pst/distances/other_distances.hpp"
#include "pst/distances/parallelize.hpp"
#include "pst/probabilistic_suffix_tree_map.hpp"

#include "io_utils.hpp"

using tree_t = pst::ProbabilisticSuffixTreeMap<seqan3::dna5>;

void prettyPrint(size_t insert_time_fst, size_t insert_time_snd, size_t find_time_fst, size_t find_time_snd, 
      size_t iterate_time, size_t dvstar_time, int items_fst, int items_snd, std::string container){
  int string_length = container.length() + 24; 
  std::cout << std::string(string_length, '-') << std::endl;
  std::cout << "|           " << container << "           |" << std::endl; 
  std::cout << std::string(string_length, '-') << std::endl; 
  // std::cout << std::string(string_length, ' ') << std::endl;
  std::cout << "    First Insert" << std::endl;  
  std::cout << "Total time : " << insert_time_fst / 1000 << " [micro sec] " << std::endl; 
  std::cout << "Sec / item : " << insert_time_fst / items_fst << " [nano sec] " << std::endl; 

  std::cout << "    Second Insert" << std::endl;  
  std::cout << "Total time : " << insert_time_snd / 1000 << " [micro sec] " << std::endl; 
  std::cout << "Sec / item : " << insert_time_snd / items_snd << " [nano sec] " << std::endl;

  std::cout << "    First Find" << std::endl;  
  std::cout << "Total time : " << find_time_fst / 1000 << " [micro sec] " << std::endl; 
  std::cout << "Sec / item : " << find_time_fst / items_fst << " [nano sec] " << std::endl;

  std::cout << "    Second Find" << std::endl;  
  std::cout << "Total time : " << find_time_snd / 1000 << " [micro sec] " << std::endl; 
  std::cout << "Sec / item : " << find_time_snd / items_snd << " [nano sec] " << std::endl;

  std::cout << "    Iterate" << std::endl;  
  std::cout << "Total time : " << iterate_time / 1000 << " [micro sec] " << std::endl; 
  std::cout << "Sec / item : " << iterate_time / (items_fst + items_snd) << " [nano sec] " << std::endl;

  std::cout << "    Dvstar" << std::endl;  
  std::cout << "Total time : " << dvstar_time / 1000 << " [micro sec] " << std::endl; 
  std::cout << "Sec / item : " << dvstar_time / (items_fst + items_snd) << " [nano sec] " << std::endl;
  std::cout << std::endl; 
  std::cout << "Items in first = " << items_fst << ", items in second = " << items_snd << std::endl;
}

void iterate_kmers_f(tree_t left, tree_t right){
  int background_order = 0; 
  double dot_product = 0.0;

  double left_norm = 0.0;
  double right_norm = 0.0;

  pst::distances::details::iterate_included_in_both<seqan3::dna5>(
      left, right, [&](auto &context, auto &left_v, auto &right_v) {
        auto [left_components, right_components] = pst::distances::details::dvstar::core_dvstar_f<seqan3::dna5>(
            left, right, background_order, context, left_v, right_v);

        for (int i = 0; i < left_components.size(); i++) {
          dot_product += left_components[i] * right_components[i];
          left_norm += std::pow(left_components[i], 2.0);
          right_norm += std::pow(right_components[i], 2.0);
        }
      });
}

std::tuple<std::vector<vlmc::VLMCKmer>, int> get_kmer_vector(std::filesystem::path path){
  std::vector<vlmc::VLMCKmer> kmers{};

  std::ifstream ifs(path, std::ios::binary);
  cereal::BinaryInputArchive archive(ifs);
  vlmc::VLMCKmer input_kmer{};

  int items = 0;

  while (ifs.peek() != EOF){
    archive(input_kmer);
    kmers.push_back(input_kmer);
    items++;
  }
  ifs.close();
  return std::make_tuple(kmers, items);
}

void run_timer(std::string container){
  std::filesystem::path path_fst{"../../../data/one_human_VLMCs/human_genome_1.bintree"};
  std::filesystem::path path_snd{"../../../data/one_human_VLMCs/human_genome_2.bintree"};

  auto fst = get_kmer_vector(path_fst);
  auto snd = get_kmer_vector(path_snd);

  int items_fst = std::get<1>(fst);
  int items_snd = std::get<1>(snd); 
  auto kmers_fst = std::get<0>(fst);
  auto kmers_snd = std::get<0>(snd);

  std::chrono::steady_clock::time_point begin_insert_fst = std::chrono::steady_clock::now();
  tree_t vlmc_fst{path_fst, 1.0}; 
  std::chrono::steady_clock::time_point end_insert_fst = std::chrono::steady_clock::now(); 
  auto insert_time_fst = std::chrono::duration_cast<std::chrono::nanoseconds>(end_insert_fst - begin_insert_fst).count();

  std::chrono::steady_clock::time_point begin_insert_snd = std::chrono::steady_clock::now();
  tree_t vlmc_snd{path_snd, 1.0}; 
  std::chrono::steady_clock::time_point end_insert_snd = std::chrono::steady_clock::now(); 
  auto insert_time_snd = std::chrono::duration_cast<std::chrono::nanoseconds>(end_insert_snd - begin_insert_snd).count();

  auto begin_find_fst = std::chrono::steady_clock::now();
  for (int i = 0; i < kmers_fst.size(); i++){
    vlmc_fst.counts.find(kmers_fst[i].to_string());
  }
  auto end_find_fst = std::chrono::steady_clock::now();
  auto find_time_fst = std::chrono::duration_cast<std::chrono::nanoseconds>(end_find_fst - begin_find_fst).count();

  auto begin_find_snd = std::chrono::steady_clock::now();
  for (int i = 0; i < kmers_snd.size(); i++){
    vlmc_snd.counts.find(kmers_snd[i].to_string());
  }
  auto end_find_snd = std::chrono::steady_clock::now();
  auto find_time_snd = std::chrono::duration_cast<std::chrono::nanoseconds>(end_find_snd - begin_find_snd).count();

  auto begin_iterate = std::chrono::steady_clock::now();
  iterate_kmers_f(vlmc_fst, vlmc_snd);
  auto end_iterate = std::chrono::steady_clock::now();
  auto iterate_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_iterate - begin_iterate).count();

  auto begin_dvstar = std::chrono::steady_clock::now();
  pst::distances::details::dvstar::core_dvstar(vlmc_fst, vlmc_snd, 0.0);
  auto end_dvstar = std::chrono::steady_clock::now();
  auto dvstar_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dvstar - begin_dvstar).count();
  
  prettyPrint(insert_time_fst, insert_time_snd, find_time_fst, find_time_snd, iterate_time, dvstar_time, items_fst, items_snd, container);
}

int main(int argc, char *argv[]){
  run_timer("Joels Hashmap");
}