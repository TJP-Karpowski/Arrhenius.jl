push!(LOAD_PATH, "../src/")

using Documenter
using Arrhenius

makedocs(
    # Explicit .html links work both on a hosted site and when a developer
    # opens docs/build/index.html directly without a local web server.
    format=Documenter.HTML(prettyurls=false),
    sitename="Arrhenius.jl",
    modules=[Arrhenius],
    remotes=nothing,
    checkdocs=:none,
    pages=[
        ##############################################
        ## MAKE SURE TO SYNC WITH docs/src/index.md ##
        ##############################################
        "Basics" => [
            "index.md",
            "install.md",
            "get_started.md",
            "equilibrium.md",
            "examples.md",
            "concepts.md"
           ],
        "tutorial.md",
        "open_source_roadmap.md",
        "solver_support.md",
        "mechanism_support.md",
        "jacobian_performance_plan.md",
        "faq.md",
        "api.md"
    ],
)

#deploydocs(;
#    repo="github.com/DENG-MIT/Arrhenius.jl",
#    devbranch="main"
#)
