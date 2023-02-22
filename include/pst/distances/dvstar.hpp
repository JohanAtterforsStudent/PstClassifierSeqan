#pragma once

#include "robin_hood.h"

#include "Eigen/Dense"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <math.h>
#include <chrono>

#include "seqan3/alphabet/concept.hpp"
#include "seqan3/alphabet/nucleotide/dna5.hpp"

#include "../probabilistic_suffix_tree_map.hpp"
#include "composition_vectors.hpp"

int nr_get_component = 0;
int nr_core_dvstar_f = 0;
int nr_contextsize_less_eq_background_order = 0;
int nr_valid_characters = 0;
int total_time_get_comp = 0;

namespace pst::distances::details::dvstar {
template <seqan3::alphabet alphabet_t>
double get_component(ProbabilisticSuffixTreeMap<alphabet_t> &tree,
                     const hashmap_value<alphabet_t> &v,
                     const std::string &context,
                     const hashmap_value<alphabet_t> &background_v,
                     const size_t char_rank) {
  nr_get_component++;
  const double background_prob =
      background_v.next_symbol_probabilities[char_rank];
  if (background_prob == 0.0) {
    return 0.0;
  } else {
    const double prob = v.next_symbol_probabilities[char_rank];
    return prob / std::sqrt(background_prob);
  }
}

template <seqan3::alphabet alphabet_t>
double
get_component(ProbabilisticSuffixTreeMap<alphabet_t> &tree,
              const hashmap_value<alphabet_t> &v, const std::string &context,
              const std::string &background_context, const size_t char_rank) {
  const double background_prob =
      tree.counts[context].next_symbol_probabilities[char_rank];
  if (background_prob == 0.0) {
    return 0.0;
  } else {
    const double prob = v.next_symbol_probabilities[char_rank];
    return prob / std::sqrt(background_prob);
  }
}

double normalise_dvstar(double dot_product, double left_norm,
                        double right_norm) {
  left_norm = std::sqrt(left_norm);
  right_norm = std::sqrt(right_norm);

  if (left_norm == 0 || right_norm == 0) {
    return 1.0;
  } else {
    double Dvstar = dot_product / (left_norm * right_norm);

    double dvstar = 0.5 * (1 - Dvstar);

    double angular_distance = 2 * std::acos(Dvstar) / M_PI;
    if (isnan(angular_distance)) {
      return 0.0;
    } else {
      return angular_distance;
    }
  }
}
template <seqan3::alphabet alphabet_t>
double get_root_count(ProbabilisticSuffixTreeMap<alphabet_t> &vlmc) {
  if (vlmc.root_state.count == 0) {
    vlmc.set_root_state();
  }
  return vlmc.root_state.count;
}

template <seqan3::alphabet alphabet_t>
std::array<std::vector<double>, 2>
core_dvstar_f(ProbabilisticSuffixTreeMap<alphabet_t> &left,
              ProbabilisticSuffixTreeMap<alphabet_t> &right,
              const size_t background_order, const std::string &context,
              const hashmap_value<alphabet_t> &left_v,
              const hashmap_value<alphabet_t> &right_v) {
  nr_core_dvstar_f++;
  if (context.size() <= background_order) {
    nr_contextsize_less_eq_background_order++;
    return {};
  }

  double dot_product = 0.0;

  double left_norm = 0.0;
  double right_norm = 0.0;

  const auto background_context =
      pst::distances::details::get_background_context(context,
                                                      background_order);

  auto left_background_v = left.counts[background_context];
  auto right_background_v = right.counts[background_context];

  std::array<std::vector<double>, 2> components{};

  for (auto &char_rank : left.valid_characters) {
    auto begin_get_comp = std::chrono::steady_clock::now();
    double left_component_value =
        get_component(left, left_v, context, left_background_v, char_rank);

    double right_component_value =
        get_component(right, right_v, context, right_background_v, char_rank);
    
    auto end_get_comp = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_get_comp - begin_get_comp).count() / 2;
    total_time_get_comp += time;

    components[0].push_back(left_component_value);
    components[1].push_back(right_component_value);
  }
  //std::cout << "nr_valid_characters : " << nr_get_component << std::endl;
  nr_valid_characters = 0;
  return components;
}

template <seqan3::alphabet alphabet_t>
inline double core_dvstar(ProbabilisticSuffixTreeMap<alphabet_t> &left,
                          ProbabilisticSuffixTreeMap<alphabet_t> &right,
                          size_t background_order) {
  double dot_product = 0.0;

  double left_norm = 0.0;
  double right_norm = 0.0;

  auto begin_it_incl = std::chrono::steady_clock::now();
  pst::distances::details::iterate_included_in_both<alphabet_t>(
      left, right, [&](auto &context, auto &left_v, auto &right_v) {
        auto [left_components, right_components] = core_dvstar_f<alphabet_t>(
            left, right, background_order, context, left_v, right_v);

        for (int i = 0; i < left_components.size(); i++) {
          dot_product += left_components[i] * right_components[i];
          left_norm += std::pow(left_components[i], 2.0);
          right_norm += std::pow(right_components[i], 2.0);
        }
      });
  auto end_it_incl = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_it_incl - begin_it_incl).count();

  std::cout << std::endl;
  std::cout << "get_component() called " << nr_get_component << std::endl;
  std::cout << "Time spent in get_component : " << total_time_get_comp << " ns" << std::endl;
  std::cout << "Avg time spent in get_component : " << total_time_get_comp / nr_get_component << " ns" << std::endl;
  std::cout << "Time spent in iterate_included_in_both : " << time << " ns" << std::endl;
  std::cout << "context.size <= background_order : " << nr_contextsize_less_eq_background_order << " times" << std::endl;
  nr_get_component = 0;
  nr_contextsize_less_eq_background_order = 0;
  total_time_get_comp = 0;
  return normalise_dvstar(dot_product, left_norm, right_norm);
}

template <seqan3::alphabet alphabet_t>
inline double
core_missing_nearest_dvstar(ProbabilisticSuffixTreeMap<alphabet_t> &left,
                            ProbabilisticSuffixTreeMap<alphabet_t> &right,
                            size_t background_order) {

  double dot_product = 0.0;
  double missing_dot_product = 0.0;

  double left_norm = 0.0;
  double right_norm = 0.0;

  double missing_left_norm = 0.0;
  double missing_right_norm = 0.0;

  double n_missing = 0.0;
  double n_contexts = 0.0;

  pst::distances::details::iterate_contexts<alphabet_t>(
      left, right,
      [&](auto &context, auto &left_v, auto &right_v) {
        auto [left_components, right_components] = core_dvstar_f<alphabet_t>(
            left, right, background_order, context, left_v, right_v);

        for (int i = 0; i < left_components.size(); i++) {
          dot_product += left_components[i] * right_components[i];
          left_norm += std::pow(left_components[i], 2.0);
          right_norm += std::pow(right_components[i], 2.0);
        }
      },
      [&](auto &context, auto &left_v) {
        auto [_ctx, right_v] = right.get_closest_state(context);

        auto [left_components, right_components] = core_dvstar_f<alphabet_t>(
            left, right, background_order, context, left_v, right_v);

        for (int i = 0; i < left_components.size(); i++) {
          dot_product += left_components[i] * right_components[i];
          left_norm += std::pow(left_components[i], 2.0);
          right_norm += std::pow(right_components[i], 2.0);
        }
      },
      [&](auto &context, auto &right_v) {
        auto [_ctx, left_v] = left.get_closest_state(context);

        auto [left_components, right_components] = core_dvstar_f<alphabet_t>(
            left, right, background_order, context, left_v, right_v);

        for (int i = 0; i < left_components.size(); i++) {
          dot_product += left_components[i] * right_components[i];
          left_norm += std::pow(left_components[i], 2.0);
          right_norm += std::pow(right_components[i], 2.0);
        }
      });

  auto dvstar_v = normalise_dvstar(dot_product + missing_dot_product,
                                   left_norm + missing_left_norm,
                                   right_norm + missing_right_norm);
  return dvstar_v;
}

template <seqan3::alphabet alphabet_t>
inline double
core_missing_penalized_dvstar(ProbabilisticSuffixTreeMap<alphabet_t> &left,
                              ProbabilisticSuffixTreeMap<alphabet_t> &right,
                              size_t background_order) {

  double dot_product = 0.0;

  double left_norm = 0.0;
  double right_norm = 0.0;

  double n_missing = 0.0;
  double n_contexts = 0.0;

  pst::distances::details::iterate_contexts<alphabet_t>(
      left, right,
      [&](auto &context, auto &left_v, auto &right_v) {
        auto [left_components, right_components] = core_dvstar_f<alphabet_t>(
            left, right, background_order, context, left_v, right_v);

        for (int i = 0; i < left_components.size(); i++) {
          dot_product += left_components[i] * right_components[i];
          left_norm += std::pow(left_components[i], 2.0);
          right_norm += std::pow(right_components[i], 2.0);
        }
        n_contexts++;
      },
      [&](auto &context, auto &left_v) {
        n_contexts++;
        n_missing++;
      },
      [&](auto &context, auto &right_v) {
        n_contexts++;
        n_missing++;
      });

  auto dvstar_v = normalise_dvstar(dot_product, left_norm, right_norm);
  double missing_frac = n_missing / n_contexts;

  return dvstar_v * 0.75 + missing_frac * 0.25;
}

} // namespace pst::distances::details::dvstar

namespace pst::distances {

template <seqan3::alphabet alphabet_t>
inline double dvstar(ProbabilisticSuffixTreeMap<alphabet_t> &left,
                     ProbabilisticSuffixTreeMap<alphabet_t> &right,
                     size_t background_order = 0) {
  return details::dvstar::core_dvstar<alphabet_t>(left, right,
                                                  background_order);
}

template <seqan3::alphabet alphabet_t>
inline double penalized_dvstar(ProbabilisticSuffixTreeMap<alphabet_t> &left,
                               ProbabilisticSuffixTreeMap<alphabet_t> &right,
                               size_t background_order = 0) {
  return details::dvstar::core_missing_penalized_dvstar<alphabet_t>(
      left, right, background_order);
}

template <seqan3::alphabet alphabet_t>
inline double nearest_dvstar(ProbabilisticSuffixTreeMap<alphabet_t> &left,
                             ProbabilisticSuffixTreeMap<alphabet_t> &right,
                             size_t background_order = 0) {
  return details::dvstar::core_missing_nearest_dvstar<alphabet_t>(
      left, right, background_order);
}

double dvstar_cpp(const std::string &left_tree_string,
                  const std::string &right_tree_string, int background_order) {
  pst::ProbabilisticSuffixTreeMap<seqan3::dna5> left_tree{left_tree_string,
                                                          1.0};
  pst::ProbabilisticSuffixTreeMap<seqan3::dna5> right_tree{right_tree_string,
                                                           1.0};

  return dvstar<seqan3::dna5>(left_tree, right_tree, background_order);
}

} // namespace pst::distances
