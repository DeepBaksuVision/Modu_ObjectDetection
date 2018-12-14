require(['gitbook'], function(gitbook) {
    gitbook.events.bind('page.change', function(e) {
      renderMathInElement(document.querySelector('.page-inner'), {
        macros: {
            "\\begin{align*}": "\\begin{aligned}",
            "\\end{align*}": "\\end{aligned}",
        },
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
        ],
      })
    })
})