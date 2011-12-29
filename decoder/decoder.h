#ifndef _DECODER_H_
#define _DECODER_H_

#include <iostream>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/program_options/variables_map.hpp>
#include <ff.h>

class SentenceMetadata;
struct Hypergraph;
struct DecoderImpl;

struct DecoderObserver {
  virtual ~DecoderObserver();
  virtual void NotifyDecodingStart(const SentenceMetadata& smeta);
  virtual void NotifySourceParseFailure(const SentenceMetadata& smeta);
  virtual void NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg);
  virtual void NotifyAlignmentFailure(const SentenceMetadata& semta);
  virtual void NotifyAlignmentForest(const SentenceMetadata& smeta, Hypergraph* hg);
  virtual void NotifyDecodingComplete(const SentenceMetadata& smeta);
};

struct Decoder {
  Decoder(int argc, char** argv);
  Decoder(std::istream* config_file);
  bool Decode(const std::string& input, DecoderObserver* observer = NULL);

  //@author FERHANTURE
  std::vector<boost::shared_ptr<FeatureFunction> > GetFFs();
  std::string GetTrans(int s);
  void NewDocument();
  void SetRuleFile(std::string f);

  void SetId(int id);
  ~Decoder();
  const boost::program_options::variables_map& GetConf() const { return conf; }

  void SetWeights(const std::vector<double>& weights);
    
  // add grammar rules (currently only supported by SCFG decoders)
  // that will be used on subsequent calls to Decode. rules should be in standard
  // text format. This function does NOT read from a file.
  void SetSupplementalGrammar(const std::string& grammar);
 private:
  boost::program_options::variables_map conf;
  boost::shared_ptr<DecoderImpl> pimpl_;
};

#endif
