#include "grammar.h"

#include <algorithm>
#include <utility>
#include <map>

#include "rule_lexer.h"
#include "filelib.h"
#include "tdict.h"

using namespace std;

const vector<TRulePtr> Grammar::NO_RULES;

RuleBin::~RuleBin() {}
GrammarIter::~GrammarIter() {}
Grammar::~Grammar() {}

bool Grammar::HasRuleForSpan(int i, int j, int distance) const {
  (void) i;
  (void) j;
  (void) distance;
  return true;  // always true by default
}

struct TextRuleBin : public RuleBin {
  int GetNumRules() const {
    return rules_.size();
  }
  TRulePtr GetIthRule(int i) const {
    return rules_[i];
  }
  void AddRule(TRulePtr t) {
    rules_.push_back(t);
  }
  int Arity() const {
    return rules_.front()->Arity();
  }
  void Dump() const {
    for (int i = 0; i < rules_.size(); ++i)
      cerr << rules_[i]->AsString() << endl;
  }
 private:
  vector<TRulePtr> rules_;
};

struct TextGrammarNode : public GrammarIter {
  TextGrammarNode() : rb_(NULL) {}
  ~TextGrammarNode() {
    delete rb_;
  }
  const GrammarIter* Extend(int symbol) const {
    map<WordID, TextGrammarNode>::const_iterator i = tree_.find(symbol);
    if (i == tree_.end()) return NULL;
    return &i->second;
  }

  const RuleBin* GetRules() const {
    if (rb_) {
      //rb_->Dump();
    }
    return rb_;
  }

  map<WordID, TextGrammarNode> tree_;
  TextRuleBin* rb_;
};

struct TGImpl {
  TextGrammarNode root_;
};

TextGrammar::TextGrammar() : max_span_(10), pimpl_(new TGImpl) {}
TextGrammar::TextGrammar(const string& file) :
    max_span_(10),
    pimpl_(new TGImpl) {
  ReadFromFile(file);
}

TextGrammar::TextGrammar(istream* in) :
    max_span_(10),
    pimpl_(new TGImpl) {
  ReadFromStream(in);
}

const GrammarIter* TextGrammar::GetRoot() const {
  return &pimpl_->root_;
}

void TextGrammar::AddRule(const TRulePtr& rule, const unsigned int ctf_level, const TRulePtr& coarse_rule) {
  if (ctf_level > 0) {
    // assume that coarse_rule is already in tree (would be safer to check)
    if (coarse_rule->fine_rules_ == 0)
      coarse_rule->fine_rules_.reset(new std::vector<TRulePtr>());
    coarse_rule->fine_rules_->push_back(rule);
    ctf_levels_ = std::max(ctf_levels_, ctf_level);
  } else if (rule->IsUnary()) {
    rhs2unaries_[rule->f().front()].push_back(rule);
    unaries_.push_back(rule);
  } else {
    TextGrammarNode* cur = &pimpl_->root_;
    for (int i = 0; i < rule->f_.size(); ++i)
      cur = &cur->tree_[rule->f_[i]];
    if (cur->rb_ == NULL)
      cur->rb_ = new TextRuleBin;
    cur->rb_->AddRule(rule);
  }
}

static void AddRuleHelper(const TRulePtr& new_rule, const unsigned int ctf_level, const TRulePtr& coarse_rule, void* extra) {
  static_cast<TextGrammar*>(extra)->AddRule(new_rule, ctf_level, coarse_rule);
}

void TextGrammar::ReadFromFile(const string& filename) {
  ReadFile in(filename);
  ReadFromStream(in.stream());
}

void TextGrammar::ReadFromStream(istream* in) {
  RuleLexer::ReadRules(in, &AddRuleHelper, this);
}

bool TextGrammar::HasRuleForSpan(int /* i */, int /* j */, int distance) const {
  return (max_span_ >= distance);
}

GlueGrammar::GlueGrammar(const string& file) : TextGrammar(file) {}

void RefineRule(TRulePtr pt, const unsigned int ctf_level){
  for (unsigned int i=0; i<ctf_level; ++i){
    TRulePtr r(new TRule(*pt));
    pt->fine_rules_.reset(new vector<TRulePtr>);
    pt->fine_rules_->push_back(r);
    pt = r;
  }
}

GlueGrammar::GlueGrammar(const string& goal_nt, const string& default_nt, const unsigned int ctf_level) {
  TRulePtr stop_glue(new TRule("[" + goal_nt + "] ||| [" + default_nt + ",1] ||| [1]"));
  AddRule(stop_glue);
  RefineRule(stop_glue, ctf_level);
  TRulePtr glue(new TRule("[" + goal_nt + "] ||| [" + goal_nt + ",1] ["+ default_nt + ",2] ||| [1] [2] ||| Glue=1"));
  AddRule(glue);
  RefineRule(glue, ctf_level);
}

bool GlueGrammar::HasRuleForSpan(int i, int /* j */, int /* distance */) const {
  return (i == 0);
}

PassThroughGrammar::PassThroughGrammar(const Lattice& input, const string& cat, const unsigned int ctf_level) :
    has_rule_(input.size() + 1) {
  for (int i = 0; i < input.size(); ++i) {
    const vector<LatticeArc>& alts = input[i];
    for (int k = 0; k < alts.size(); ++k) {
      const int j = alts[k].dist2next + i;
      has_rule_[i].insert(j);
      const string& src = TD::Convert(alts[k].label);
      TRulePtr pt(new TRule("[" + cat + "] ||| " + src + " ||| " + src + " ||| PassThrough=1"));
      pt->a_.push_back(AlignmentPoint(0,0));
      AddRule(pt);
      RefineRule(pt, ctf_level);
    }
  }
}

bool PassThroughGrammar::HasRuleForSpan(int i, int j, int /* distance */) const {
  const set<int>& hr = has_rule_[i];
  if (i == j) { return !hr.empty(); }
  return (hr.find(j) != hr.end());
}
