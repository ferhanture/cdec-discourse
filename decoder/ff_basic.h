#ifndef _FF_BASIC_H_
#define _FF_BASIC_H_

#include "ff.h"
#include <stdint.h>
#include <vector>
#include <cstring>
#include "fdict.h"
#include "hg.h"
#include "feature_vector.h"
#include "value_array.h"
#include <iostream>
#include "fast_lexical_cast.hpp"
#include <stdexcept>
#include <map>
#include <boost/foreach.hpp>
#include "tdict.h"
#include "hg.h"
#include "filelib.h"
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string.hpp>

// word penalty feature, for each word on the E side of a rule,
// add value_
class WordPenalty : public FeatureFunction {
 public:
  WordPenalty(const std::string& param);
  static std::string usage(bool p,bool d) {
    return usage_helper("WordPenalty","","number of target words (local feature)",p,d);
  }
 protected:
  virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                     const HG::Edge& edge,
                                     const std::vector<const void*>& ant_contexts,
                                     SparseVector<double>* features,
                                     SparseVector<double>* estimated_features,
                                     void* context) const;
 private:
  const int fid_;
  const double value_;
};

class Discourse : public FeatureFunction {
	typedef std::map<std::string, float> MapType;
	
public:
	Discourse(const std::string& param);
	static std::string usage(bool p,bool d) {
		return usage_helper("Discourse","","award one translation per discourse",p,d);
	}
	bool rule_feature() const { return true; }
	void PrepareForInput();
	
	void set_dfs(std::map<std::string,int> df) { dfs = df; }
    
    
	void load_freqs(int docid, std::string rulesdir);
	void load_freqs(std::string rulesfile);
    const static bool DEBUG=false;
	
	void set_df_numdocs(int n) {	num_docs = n; }
//    void set_isdisc(bool is0, bool is1, bool is2) {	
//        if(is0){
//            isDisc.push_back(true);
//        }else{
//            isDisc.push_back(false);
//        }
//        if(is1){
//            isDisc.push_back(true);
//        }else{
//            isDisc.push_back(false);
//        }
//        if(is2){
//            isDisc.push_back(true);
//        }else{
//            isDisc.push_back(false);
//        }
//    }

protected:
	virtual void PrepareForInput(const SentenceMetadata& smeta);
	
	virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
									   const Hypergraph::Edge& edge,
									   const std::vector<const void*>& ant_contexts,
									   FeatureVector* features,
									   FeatureVector* estimated_features,
									   void* context) const;
private:
	float compute_fval(int mode, std::string rule, MapType rule_freq) const;
	float bm25(int tf) const;
	float bm25(int tf, int df) const;
	std::vector<std::string> parse_rule_into_vector(const int mode, const std::string &line) const;
	bool is_glue(const std::string &line) const;
    
	std::vector<std::string> split(const std::string& s, const std::string& delim, const bool keep_empty) const;
	
	std::vector<WordID> fids_;
    
    std::vector<bool> isDisc;
	std::vector<MapType> rule_freq;
	std::map<std::string,int> dfs;
	int num_docs, fid_;
};

class SourceWordPenalty : public FeatureFunction {
 public:
  SourceWordPenalty(const std::string& param);
  static std::string usage(bool p,bool d) {
    return usage_helper("SourceWordPenalty","","number of source words (local feature, and meaningless except when input has non-constant number of source words, e.g. segmentation/morphology/speech recognition lattice)",p,d);
  }
 protected:
  virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                     const HG::Edge& edge,
                                     const std::vector<const void*>& ant_contexts,
                                     SparseVector<double>* features,
                                     SparseVector<double>* estimated_features,
                                     void* context) const;
 private:
  const int fid_;
  const double value_;
};

#define DEFAULT_MAX_ARITY 9
#define DEFAULT_MAX_ARITY_STRINGIZE(x) #x
#define DEFAULT_MAX_ARITY_STRINGIZE_EVAL(x) DEFAULT_MAX_ARITY_STRINGIZE(x)
#define DEFAULT_MAX_ARITY_STR DEFAULT_MAX_ARITY_STRINGIZE_EVAL(DEFAULT_MAX_ARITY)

class ArityPenalty : public FeatureFunction {
 public:
  ArityPenalty(const std::string& param);
  static std::string usage(bool p,bool d) {
    return usage_helper("ArityPenalty","[MaxArity(default " DEFAULT_MAX_ARITY_STR ")]","Indicator feature Arity_N=1 for rule of arity N (local feature).  0<=N<=MaxArity(default " DEFAULT_MAX_ARITY_STR ")",p,d);
  }

 protected:
  virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                     const HG::Edge& edge,
                                     const std::vector<const void*>& ant_contexts,
                                     SparseVector<double>* features,
                                     SparseVector<double>* estimated_features,
                                     void* context) const;
 private:
  std::vector<WordID> fids_;
  const double value_;
};

#endif
