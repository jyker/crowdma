
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/jyker/crowdma">
  </a>

  <p align="center">
    Crowdsourcing Malware Family Annotation: Joint Class-determined Tag Extraction and Weakly-tagged Sample Inference
    <br />

  </p>
</div>


<!-- ABOUT THE PROJECT -->
## Abstract

Anti-malware engines report malware labels to detail malice, typically including tags of family, behavior, and platform classes. This capability has been heavily used by the security community to annotate malware families and build reference datasets, which is referred to as crowdsourcing malware family annotation. However, how to associate tags with their corresponding classes in chaotic malware labels (extract class-determined tags) and how to infer ground truth for weakly-tagged samples that hold controversial tags remain open problems. In this paper, we present a novel annotation pipeline to advance further, which includes an incremental parsing scheme and a maximum likelihood estimation scheme. The incremental parsing scheme treats behavior and platform tags as locators and achieves incremental parsing by introducing and iterating the following two algorithms: location ﬁrst search, which hits family tags using locators, and co-occurrence ﬁrst search, which ﬁnds new locators by family tags. The maximum likelihood estimating scheme models an engine’s ability to identify different families as a confusion matrix and introduces an expectation-maximization algorithm to estimate the matrix, as well as the unknown truth of samples. Experiments across four benchmark datasets indicate that our pipeline outperforms existing work, improving label-level parsing accuracy by an average of 29%, and improving inferring accuracy on weakly-tagged samples by an average of 9%. Our pipeline decouples parsing and inferring, which would pave the way for research on crowdsourcing malware family annotation.

## License

Distributed under the MIT License.