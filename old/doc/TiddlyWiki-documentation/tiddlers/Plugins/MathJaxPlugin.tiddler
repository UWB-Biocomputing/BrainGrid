/***
|''Name:''|MathJaxPlugin|
|''Description:''|Enable LaTeX formulas for TiddlyWiki|
|''Version:''|1.0.1|
|''Date:''|Feb 11, 2012|
|''Source:''|http://www.guyrutenberg.com/2011/06/25/latex-for-tiddlywiki-a-mathjax-plugin|
|''Author:''|Guy Rutenberg|
|''License:''|[[BSD open source license]]|
|''~CoreVersion:''|2.5.0|
 
!! Changelog
!!! 1.0.1 Feb 11, 2012
* Fixed interoperability with TiddlerBarPlugin
!! How to Use
Currently the plugin supports the following delemiters:
* """\(""".."""\)""" - Inline equations
* """$$""".."""$$""" - Displayed equations
* """\[""".."""\]""" - Displayed equations
!! Demo
This is an inline equation \(P(E)   = {n \choose k} p^k (1-p)^{ n-k}\) and this is a displayed equation:
\[J_\alpha(x) = \sum_{m=0}^\infty \frac{(-1)^m}{m! \, \Gamma(m + \alpha + 1)}{\left({\frac{x}{2}}\right)}^{2 m + \alpha}\]
This is another displayed equation $$e=mc^2$$
!! Code
***/
//{{{
config.extensions.MathJax = {
  mathJaxScript : "http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML",
  // uncomment the following line if you want to access MathJax using SSL
  // mathJaxScript : "https://d3eoax9i5htok0.cloudfront.net/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML",
  displayTiddler: function(TiddlerName) {
    config.extensions.MathJax.displayTiddler_old.apply(this, arguments);
    MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
  }
};
 
jQuery.getScript(config.extensions.MathJax.mathJaxScript, function(){
    MathJax.Hub.Config({
      extensions: ["tex2jax.js"],
      "HTML-CSS": { scale: 100 }
    });
 
    MathJax.Hub.Startup.onload();
    config.extensions.MathJax.displayTiddler_old = story.displayTiddler;
    story.displayTiddler = config.extensions.MathJax.displayTiddler;
});
 
config.formatters.push({
	name: "mathJaxFormula",
	match: "\\\\\\[|\\$\\$|\\\\\\(",
	//lookaheadRegExp: /(?:\\\[|\$\$)((?:.|\n)*?)(?:\\\]|$$)/mg,
	handler: function(w)
	{
		switch(w.matchText) {
		case "\\[": // displayed equations
			this.lookaheadRegExp = /\\\[((?:.|\n)*?)(\\\])/mg;
			break;
		case "$$": // inline equations
			this.lookaheadRegExp = /\$\$((?:.|\n)*?)(\$\$)/mg;
			break;
		case "\\(": // inline equations
			this.lookaheadRegExp = /\\\(((?:.|\n)*?)(\\\))/mg;
			break;
		default:
			break;
		}
		this.lookaheadRegExp.lastIndex = w.matchStart;
		var lookaheadMatch = this.lookaheadRegExp.exec(w.source);
		if(lookaheadMatch && lookaheadMatch.index == w.matchStart) {
			createTiddlyElement(w.output,"span",null,null,lookaheadMatch[0]);
			w.nextMatch = this.lookaheadRegExp.lastIndex;
		}
	}
});
//}}}
