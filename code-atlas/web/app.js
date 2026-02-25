(function () {
  var app = document.getElementById("app");
  var navTabs = Array.from(document.querySelectorAll(".nav-tabs .tab"));

  var model = {
    index: null,
    snippets: [],
    snippetsById: {},
    symbols: [],
    grepHits: [],
    files: [],
    filesByKey: {},
    filesLoaded: false,
    filesPromise: null,
    commits: null,
    reportMd: ""
  };

  var state = {
    view: "home",
    selectedSeamId: null,
    seamBucket: "upstream-vllm",
    focusSnippetId: "",
    searchQuery: "",
    searchType: "all",
    diffSnippetA: "",
    diffSnippetB: "",
    diffFileA: "",
    diffFileB: "",
    showContext: {},
    showFile: {},
    showDrawerContent: {},
    expandSnippet: {},
    starred: new Set()
  };

  var STAR_KEY = "codeAtlasStarredSnippets";

  function esc(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function loadStarred() {
    try {
      var raw = localStorage.getItem(STAR_KEY);
      if (!raw) return;
      var arr = JSON.parse(raw);
      if (Array.isArray(arr)) {
        state.starred = new Set(arr);
      }
    } catch (_e) {
      state.starred = new Set();
    }
  }

  function persistStarred() {
    localStorage.setItem(STAR_KEY, JSON.stringify(Array.from(state.starred)));
  }

  function link(url, label) {
    return '<a href="' + esc(url) + '" target="_blank" rel="noreferrer">' + esc(label) + "</a>";
  }

  function seamPills(seamIds) {
    return (seamIds || [])
      .map(function (sid) {
        return (
          '<button class="pill" data-action="open-seam" data-seam="' +
          esc(sid) +
          '">' +
          esc(sid) +
          "</button>"
        );
      })
      .join(" ");
  }

  function bulletList(items, className) {
    if (!items || !items.length) return '<p class="small">No items.</p>';
    return (
      '<ul class="' +
      esc(className || "mini-list") +
      '">' +
      items
        .map(function (item) {
          return "<li>" + esc(item) + "</li>";
        })
        .join("") +
      "</ul>"
    );
  }

  function storyKindLabel(kind) {
    return (
      {
        contract: "Contract",
        call_site: "Call site",
        data_structure: "Data structure",
        lifecycle: "Lifecycle",
        divergence: "Divergence",
        example: "Example"
      }[kind] || "Evidence"
    );
  }

  function bucketForSnippet(snippet) {
    if (snippet.repo === "vllm") return "upstream-vllm";
    if (snippet.repo === "torch-spyre") return "torch-spyre";
    if (snippet.repo === "vllm-spyre") {
      if (snippet.file_path.indexOf("vllm_spyre_next/") === 0) {
        return "vllm-spyre-next";
      }
      return "vllm-spyre";
    }
    return "other";
  }

  function bucketLabel(bucket) {
    return {
      "upstream-vllm": "Upstream vLLM",
      "vllm-spyre": "vllm-spyre today",
      "vllm-spyre-next": "vllm-spyre-next",
      "torch-spyre": "torch-spyre"
    }[bucket] || bucket;
  }

  function setActiveTab() {
    navTabs.forEach(function (tab) {
      tab.classList.toggle("active", tab.dataset.view === state.view);
    });
  }

  function indexFiles(files) {
    var byKey = {};
    (files || []).forEach(function (f) {
      if (f && f.file_key) byKey[f.file_key] = f;
    });
    model.files = files || [];
    model.filesByKey = byKey;
    model.filesLoaded = true;
  }

  function ensureFilesLoaded() {
    if (model.filesLoaded) {
      return Promise.resolve(model.files);
    }
    if (model.filesPromise) {
      return model.filesPromise;
    }

    model.filesPromise = fetch("./data/files.json")
      .then(function (r) {
        if (!r.ok) throw new Error("Failed to load files.json (" + r.status + ")");
        return r.json();
      })
      .then(function (files) {
        indexFiles(files);
        return model.files;
      })
      .finally(function () {
        model.filesPromise = null;
      });

    return model.filesPromise;
  }

  function codeTable(lines, language, hlStart, hlEnd) {
    var rows = lines
      .map(function (item) {
        var lineNo = item.line;
        var text = item.text;
        var cls = lineNo >= hlStart && lineNo <= hlEnd ? "hl" : "";
        var highlighted = Prism.highlight(text, Prism.languages[language] || null, language);
        return (
          '<tr class="' +
          cls +
          '"><td class="no">' +
          lineNo +
          '</td><td class="code"><code class="language-' +
          language +
          '">' +
          highlighted +
          "</code></td></tr>"
        );
      })
      .join("");

    return '<div class="code-wrap"><table class="code-table"><tbody>' + rows + "</tbody></table></div>";
  }

  function snippetCard(snippet, options) {
    options = options || {};
    var storyMode = !!options.storyMode;
    var storyIndex = options.storyIndex || 0;
    var isStarred = state.starred.has(snippet.id);
    var contextVisible = !!state.showContext[snippet.id];
    var fileVisible = !!state.showFile[snippet.id];
    var bucket = bucketForSnippet(snippet);
    var isExpanded = !!state.expandSnippet[snippet.id];
    var previewLimit = storyMode ? 140 : 80;
    var lineCount = (snippet.lines || []).length;
    var isTruncated = !isExpanded && lineCount > previewLimit;
    var displayLines = isTruncated ? snippet.lines.slice(0, previewLimit) : snippet.lines;

    var refBadges = "";
    if (snippet.head_in_origin_refs === false || snippet.permalink_mode === "sha_not_in_origin_refs") {
      refBadges += '<span class="badge warn-badge">sha not in local origin refs</span>';
    }
    if (snippet.repo_dirty) {
      refBadges += '<span class="badge warn-badge">dirty tree</span>';
    }
    if (snippet.source_mode && snippet.source_mode !== "head_commit") {
      refBadges += '<span class="badge warn-badge">source: ' + esc(snippet.source_mode) + "</span>";
    }

    var head =
      '<span class="badge repo-' +
      esc(snippet.repo) +
      '">' +
      esc(bucketLabel(bucket)) +
      "</span>" +
      '<span class="badge">' +
      esc(snippet.file_path) +
      "</span>" +
      '<span class="badge">' +
      esc(snippet.commit_short) +
      "</span>" +
      '<span class="badge">' +
      esc(snippet.head_ref || "HEAD") +
      "</span>" +
      '<span class="badge">' +
      esc((snippet.evidence_tier || "unknown").toUpperCase()) +
      "</span>" +
      '<span class="badge">' +
      esc(storyKindLabel(snippet.story_kind)) +
      "</span>" +
      '<span class="badge">target #' +
      esc(snippet.target_index || "") +
      "</span>" +
      '<span class="badge">L' +
      snippet.start_line +
      "-L" +
      snippet.end_line +
      "</span>" +
      refBadges;

    var permalinkPrimaryLabel = "Permalink (sha)";
    var extraPermalink = "";
    if (snippet.permalink_branch && snippet.permalink_sha && snippet.permalink_branch !== snippet.permalink_sha) {
      var secondaryLabel = "Permalink (branch)";
      var secondaryUrl = snippet.permalink_branch;
      extraPermalink =
        '<a href="' +
        esc(secondaryUrl) +
        '" target="_blank" rel="noreferrer">' +
        esc(secondaryLabel) +
        "</a>";
    }

    var actions =
      '<a href="' +
      esc(snippet.permalink) +
      '" target="_blank" rel="noreferrer">' +
      esc(permalinkPrimaryLabel) +
      "</a>" +
      extraPermalink +
      '<button data-action="copy-snippet" data-snippet="' +
      esc(snippet.id) +
      '">Copy code</button>' +
      '<button data-action="toggle-context" data-snippet="' +
      esc(snippet.id) +
      '">' +
      (contextVisible ? "Hide" : "Show") +
      " context ±50</button>" +
      '<button data-action="toggle-file" data-snippet="' +
      esc(snippet.id) +
      '">' +
      (fileVisible ? "Hide" : "Show") +
      " full file</button>" +
      (lineCount > previewLimit
        ? '<button data-action="toggle-snippet-size" data-snippet="' +
          esc(snippet.id) +
          '">' +
          (isExpanded ? "Show less" : "Show full snippet (" + lineCount + " lines)") +
          "</button>"
        : "") +
      '<button class="' +
      (isStarred ? "starred" : "") +
      '" data-action="toggle-star" data-snippet="' +
      esc(snippet.id) +
      '">' +
      (isStarred ? "Unstar" : "Star") +
      "</button>";

    var sourceRefShort = snippet.source_ref ? String(snippet.source_ref).slice(0, 10) : "";
    var layer1 =
      '<div class="snippet-note">' +
      (storyMode
        ? '<p class="story-kicker">Story snippet ' +
          esc(storyIndex) +
          " · " +
          esc(storyKindLabel(snippet.story_kind)) +
          "</p>"
        : "") +
      '<p><strong>What this proves:</strong> ' +
      esc(snippet.story_proves || snippet.target_takeaway || "No explicit proof note.") +
      "</p>" +
      '<p><strong>Boundary:</strong> ' +
      esc(snippet.story_boundary || snippet.target_role || "Boundary note unavailable.") +
      "</p>" +
      "</div>";

    var implications = "";
    var hasImplications = (snippet.story_implications || []).length || (snippet.story_risks || []).length || snippet.story_inference;
    if (hasImplications) {
      implications =
        "<details class=\"snippet-implications\"><summary>Implications and risks</summary>" +
        ((snippet.story_implications || []).length
          ? '<p><strong>Implications:</strong></p>' + bulletList(snippet.story_implications, "mini-list")
          : "") +
        ((snippet.story_risks || []).length
          ? '<p><strong>Risks/failure modes:</strong></p>' + bulletList(snippet.story_risks, "mini-list")
          : "") +
        (snippet.story_inference
          ? '<p><strong>Inference:</strong> ' + esc(snippet.story_inference) + "</p>"
          : "") +
        "</details>";
    }

    var evidenceMeta =
      '<div class="snippet-note">' +
      '<p><strong>Why this snippet is shown:</strong> ' +
      esc(snippet.selection_reason || "selection rationale unavailable") +
      "</p>" +
      '<p><strong>Target:</strong> ' +
      esc(snippet.target_id || "") +
      " · kind=" +
      esc(snippet.evidence_kind || "unknown") +
      " · anchor=" +
      esc(snippet.anchor_type || snippet.extractor || "unknown") +
      "</p>" +
      '<p><strong>Source blob:</strong> ' +
      esc(snippet.source_mode || "unknown") +
      (sourceRefShort ? " @ " + esc(sourceRefShort) : "") +
      "</p>" +
      (snippet.target_role
        ? '<p><strong>Role in seam:</strong> ' + esc(snippet.target_role) + "</p>"
        : "") +
      (snippet.target_note
        ? '<p><strong>Interpretation:</strong> ' + esc(snippet.target_note) + "</p>"
        : "") +
      (snippet.target_takeaway
        ? '<p><strong>Expected takeaway:</strong> ' + esc(snippet.target_takeaway) + "</p>"
        : "") +
      (snippet.target_compare
        ? '<p><strong>Comparison cue:</strong> ' + esc(snippet.target_compare) + "</p>"
        : "") +
      ((snippet.target_checklist || []).length
        ? '<div><p><strong>Verification checklist:</strong></p>' +
          bulletList(snippet.target_checklist, "mini-list") +
          "</div>"
        : "") +
      "</div>";

    var body =
      codeTable(displayLines, snippet.language, snippet.start_line, snippet.end_line) +
      (isTruncated
        ? '<p class="small code-truncate-note">Showing first ' +
          previewLimit +
          " of " +
          lineCount +
          " lines. Use \"Show full snippet\" for the full code block.</p>"
        : "");

    var context = "";
    if (contextVisible) {
      context =
        '<div class="card"><p class="small">Context lines ' +
        snippet.context_start_line +
        "-" +
        snippet.context_end_line +
        "</p>" +
        codeTable(
          snippet.context_lines,
          snippet.language,
          snippet.start_line,
          snippet.end_line
        ) +
        "</div>";
    }

    var fullFile = "";
    if (fileVisible) {
      var fileKey = snippet.repo + "::" + snippet.file_path;
      var file = model.filesByKey[fileKey];
      if (file) {
        fullFile =
          '<div class="card"><p class="small">Full file · ' +
          esc(file.file_path) +
          " · " +
          file.total_lines +
          " lines</p>" +
          codeTable(file.lines, file.language, snippet.start_line, snippet.end_line) +
          "</div>";
      } else {
        fullFile =
          '<div class="card"><p class="small">Full file index is loading. Please wait...</p></div>';
      }
    }

    return (
      '<article class="snippet-card" id="snippet-' +
      esc(snippet.id) +
      '">' +
      '<header class="snippet-head">' +
      head +
      "</header>" +
      '<div class="snippet-actions">' +
      actions +
      "</div>" +
      layer1 +
      implications +
      evidenceMeta +
      '<div class="snippet-body">' +
      body +
      "</div>" +
      '<div class="snippet-extra">' +
      context +
      fullFile +
      "</div>" +
      "</article>"
    );
  }

  function seamById(id) {
    return (model.index.seams || []).find(function (s) {
      return s.id === id;
    });
  }

  function snippetById(id) {
    return model.snippetsById[id];
  }

  function openSeam(seamId, maybeSnippetId) {
    state.view = "seams";
    state.selectedSeamId = seamId;
    state.focusSnippetId = maybeSnippetId || "";
    render();
  }

  function homeView() {
    var fpCards = (model.index.first_principles || [])
      .map(function (fp) {
        return (
          '<article class="card"><p class="kicker">' +
          esc(fp.title) +
          "</p><p>" +
          esc(fp.summary) +
          '</p><div class="row">' +
          seamPills(fp.drill_seams || []) +
          "</div></article>"
        );
      })
      .join("");

    var howToReadCards = (model.index.how_to_read || [])
      .map(function (step, idx) {
        return (
          '<article class="card"><p class="kicker">Step ' +
          (idx + 1) +
          "</p><h4>" +
          esc(step.title || "Step") +
          "</h4><p>" +
          esc(step.details || "") +
          '</p><div class="row">' +
          seamPills(step.seams || []) +
          "</div></article>"
        );
      })
      .join("");

    var layerCards = (model.index.architecture_layers || [])
      .map(function (layer) {
        return (
          '<article class="card"><p class="kicker">' +
          esc(layer.title || "Layer") +
          "</p><p>" +
          esc(layer.summary || "") +
          '</p><p class="small"><strong>Verify:</strong></p>' +
          bulletList(layer.what_to_verify || [], "mini-list") +
          '<div class="row">' +
          seamPills(layer.seams || []) +
          "</div></article>"
        );
      })
      .join("");

    var legendRows = (model.index.badge_legend || [])
      .map(function (entry) {
        return (
          "<tr><td><strong>" +
          esc(entry.label || "") +
          "</strong></td><td>" +
          esc(entry.meaning || "") +
          "</td></tr>"
        );
      })
      .join("");

    var caveats = bulletList(model.index.global_caveats || [], "mini-list");

    var glossaryRows = (model.index.glossary || [])
      .map(function (entry) {
        var seams = seamPills(entry.seams || []);
        return (
          '<div class="result-item"><strong>' +
          esc(entry.term || "") +
          ":</strong> " +
          esc(entry.definition || "") +
          (seams ? '<div class="row">' + seams + "</div>" : "") +
          "</div>"
        );
      })
      .join("");

    var extractionPolicy = model.index.extraction_policy || {};
    var anchorTypes = (extractionPolicy.anchor_types || [])
      .map(function (x) {
        return "<code>" + esc(x) + "</code>";
      })
      .join(", ");
    var quality = model.index.quality_report || {};
    var qagg = quality.aggregate || {};
    var qWarnings = (quality.warnings || [])
      .slice(0, 10)
      .map(function (w) {
        return "<li>" + esc(w) + "</li>";
      })
      .join("");
    var storyLegendRows = (model.index.story_kind_legend || [])
      .map(function (entry) {
        return (
          "<tr><td><strong>" +
          esc(entry.label || entry.kind || "") +
          "</strong></td><td>" +
          esc(entry.description || "") +
          "</td></tr>"
        );
      })
      .join("");

    var commits = model.commits.repos;
    var repoRows = Object.keys(commits)
      .sort()
      .map(function (name) {
        var r = commits[name];
        var originStatus = "origin refs unknown";
        if (r.head_in_origin_refs === true) originStatus = "sha in local origin refs";
        else if (r.head_in_origin_refs === false) originStatus = "sha missing from local origin refs";
        var dirtyStatus = r.is_dirty ? "dirty" : "clean";
        return (
          "<tr><td><strong>" +
          esc(name) +
          "</strong></td><td>" +
          esc(r.head_short) +
          "</td><td>" +
          esc(r.head_ref || "HEAD") +
          "</td><td>" +
          esc(dirtyStatus) +
          "</td><td>" +
          esc(originStatus) +
          "</td><td>" +
          link(r.github_base, "repo") +
          "</td></tr>"
        );
      })
      .join("");

    return (
      '<section class="card"><h3>Atlas Purpose</h3><p>' +
      esc(model.index.atlas_purpose || "Architecture-to-code walkthrough with pinned refs.") +
      '</p><p class="small"><strong>Extraction policy:</strong> regex allowed=' +
      esc(String(extractionPolicy.allow_regex_targets)) +
      ", strict target matches=" +
      esc(String(extractionPolicy.strict_target_matches)) +
      ". Anchor types: " +
      anchorTypes +
      "</p><p class=\"small\"><strong>Quality snapshot:</strong> seams=" +
      esc(qagg.seam_count || 0) +
      ", mean score=" +
      esc(typeof qagg.mean_of_means === "number" ? qagg.mean_of_means.toFixed(2) : "n/a") +
      ", full-required-matches=" +
      esc(qagg.seams_with_full_required_matches || 0) +
      "/" +
      esc(qagg.seam_count || 0) +
      ", upstream-anchored=" +
      esc(qagg.seams_with_upstream_anchor || 0) +
      "/" +
      esc(qagg.seam_count || 0) +
      ", story-budget(3-5)=" +
      esc(qagg.seams_with_story_budget_3_to_5 || 0) +
      "/" +
      esc(qagg.seam_count || 0) +
      ", completeness=" +
      esc(typeof qagg.mean_completeness === "number" ? qagg.mean_completeness.toFixed(2) : "n/a") +
      "</p>" +
      (qWarnings
        ? '<p class="small"><strong>Top warnings:</strong></p><ul class="mini-list">' + qWarnings + "</ul>"
        : "") +
      "</section>" +
      '<div class="grid two">' +
      '<section class="card"><h3>First-principles walkthrough</h3><div class="grid">' +
      fpCards +
      "</div></section>" +
      '<section class="card"><h3>Dataset status</h3><p><strong>Seams:</strong> ' +
      model.index.seam_count +
      "<br><strong>Snippets:</strong> " +
      model.index.snippet_count +
      "<br><strong>Symbols:</strong> " +
      model.symbols.length +
      "<br><strong>Grep hits:</strong> " +
      model.grepHits.length +
      '</p><table class="code-table"><tbody><tr><td><strong>repo</strong></td><td><strong>sha</strong></td><td><strong>ref</strong></td><td><strong>tree</strong></td><td><strong>origin refs</strong></td><td><strong>link</strong></td></tr>' +
      repoRows +
      "</tbody></table></section></div>" +
      '<section class="card"><h3>How to read this atlas</h3><div class="grid">' +
      howToReadCards +
      "</div></section>" +
      '<div class="grid two">' +
      '<section class="card"><h3>Architecture layers</h3><div class="grid">' +
      layerCards +
      "</div></section>" +
      '<section class="card"><h3>Global caveats</h3>' +
      caveats +
      "</section></div>" +
      '<div class="grid two">' +
      '<section class="card"><h3>Badge legend</h3><table class="code-table"><tbody>' +
      legendRows +
      "</tbody></table></section>" +
      '<section class="card"><h3>Story role legend</h3><table class="code-table"><tbody>' +
      (storyLegendRows || "<tr><td>No story legend entries.</td><td></td></tr>") +
      "</tbody></table></section></div>" +
      '<div class="grid two">' +
      '<section class="card"><h3>Glossary</h3>' +
      (glossaryRows || '<p class="small">No glossary entries yet.</p>') +
      "</section></div>"
    );
  }

  function seamMapView() {
    var seams = model.index.seams || [];
    if (!state.selectedSeamId && seams.length) {
      state.selectedSeamId = seams[0].id;
    }

    var list = seams
      .map(function (seam) {
        var active = seam.id === state.selectedSeamId ? "active" : "";
        return (
          '<button class="seam-node ' +
          active +
          '" data-action="open-seam" data-seam="' +
          esc(seam.id) +
          '"><strong>' +
          esc(seam.title) +
          "</strong><br><span class=\"small\">" +
          esc(seam.id) +
          " · " +
          seam.snippet_ids.length +
          " snippets</span></button>"
        );
      })
      .join("");

    var seam = seamById(state.selectedSeamId);
    if (!seam) {
      return '<article class="card">No seam selected.</article>';
    }
    var tags = (seam.tags || [])
      .map(function (t) {
        return '<span class="pill">' + esc(t) + "</span>";
      })
      .join(" ");

    var claimsHtml = (seam.claims || [])
      .map(function (claim) {
        return "<li>" + esc(claim) + "</li>";
      })
      .join("");

    var recommendation = seam.recommendation || {};
    var recommendationHtml =
      '<article class="card recommendation-box"><h4>Recommendation</h4>' +
      (recommendation.now
        ? "<p><strong>Now:</strong> " + esc(recommendation.now) + "</p>"
        : "") +
      (recommendation.depends_on
        ? "<p><strong>Dependency:</strong> " + esc(recommendation.depends_on) + "</p>"
        : "") +
      (recommendation.success_test
        ? "<p><strong>Success test:</strong> " + esc(recommendation.success_test) + "</p>"
        : "") +
      "</article>";

    var completeness = seam.completeness || { items: [] };
    var completenessRows = (completeness.items || [])
      .map(function (item) {
        return (
          "<tr><td>" +
          esc(item.ok ? "yes" : "no") +
          "</td><td>" +
          esc(item.label || "") +
          "</td></tr>"
        );
      })
      .join("");

    var storySnippets = (seam.story_snippet_ids || [])
      .map(function (sid) {
        return snippetById(sid);
      })
      .filter(Boolean);
    if (!storySnippets.length) {
      storySnippets = seam.snippet_ids
        .slice(0, 5)
        .map(function (sid) {
          return snippetById(sid);
        })
        .filter(Boolean);
    }
    var storyHtml = storySnippets.length
      ? storySnippets
          .map(function (sn, idx) {
            return snippetCard(sn, { storyMode: true, storyIndex: idx + 1 });
          })
          .join("")
      : '<article class="card"><p>No story snippets selected yet for this seam.</p></article>';

    var compareRows = (seam.compare_rows || [])
      .map(function (row) {
        var open = row.story_snippet_id
          ? '<button class="btn-inline" data-action="open-snippet" data-snippet="' +
            esc(row.story_snippet_id) +
            '">Open snippet</button>'
          : '<span class="small">No story snippet in this bucket.</span>';
        return (
          "<tr><td><strong>" +
          esc(bucketLabel(row.bucket)) +
          "</strong></td><td>" +
          (row.note ? esc(row.note) : '<span class="small">Target shape only; no concrete snippet selected.</span>') +
          "</td><td>" +
          open +
          "</td></tr>"
        );
      })
      .join("");

    var buckets = ["upstream-vllm", "vllm-spyre", "vllm-spyre-next", "torch-spyre"];
    var drawerIds = seam.evidence_drawer_snippet_ids || [];
    var drawerLoaded = !!state.showDrawerContent[seam.id];
    var tabBar = buckets
      .map(function (bucket) {
        var active = state.seamBucket === bucket ? "active" : "";
        var count = drawerIds.filter(function (sid) {
          var sn = snippetById(sid);
          return sn && bucketForSnippet(sn) === bucket;
        }).length;
        return (
          '<button class="repo-tab ' +
          active +
          '" data-action="set-bucket" data-bucket="' +
          bucket +
          '">' +
          esc(bucketLabel(bucket)) +
          " (" +
          count +
          ")</button>"
        );
      })
      .join("");

    var drawerSnippets = drawerIds
      .map(function (sid) {
        return snippetById(sid);
      })
      .filter(Boolean)
      .filter(function (sn) {
        return bucketForSnippet(sn) === state.seamBucket;
      });
    var drawerHtml = "";
    if (!drawerLoaded) {
      drawerHtml =
        '<article class="card"><p class="small">Drawer snippets are not rendered yet to keep seam pages fast.</p>' +
        '<button class="btn" data-action="load-drawer" data-seam="' +
        esc(seam.id) +
        '">Load evidence drawer snippets (' +
        drawerIds.length +
        ")</button></article>";
    } else {
      drawerHtml = drawerSnippets.length
        ? drawerSnippets.map(function (sn) {
            return snippetCard(sn, { storyMode: false });
          }).join("")
        : '<article class="card"><p class="small">No additional evidence snippets for this bucket.</p></article>';
    }

    var evidenceStats = seam.evidence_stats || {};
    var extractorStats = evidenceStats.extractors || {};
    var extractorRows = Object.keys(extractorStats)
      .sort()
      .map(function (k) {
        return "<tr><td><strong>" + esc(k || "unknown") + "</strong></td><td>" + esc(extractorStats[k]) + "</td></tr>";
      })
      .join("");
    var meanScore =
      typeof evidenceStats.mean_score === "number" ? evidenceStats.mean_score.toFixed(2) : "n/a";

    var targetPlanRows = (seam.target_plan || [])
      .map(function (t) {
        var rowCls = t.required && t.status !== "matched" ? " class=\"warn-row\"" : "";
        var openBtn = "";
        if (t.snippet_ids && t.snippet_ids.length) {
          openBtn =
            '<button class="btn-inline" data-action="open-snippet" data-snippet="' +
            esc(t.snippet_ids[0]) +
            '">Open</button>';
        }
        return (
          "<tr" +
          rowCls +
          "><td>" +
          esc(t.target_index) +
          "</td><td><code>" +
          esc(t.target_id || "") +
          "</code></td><td>" +
          esc(t.story_label || t.story_kind || "") +
          "</td><td>" +
          esc(t.evidence_kind || "") +
          "</td><td>" +
          esc(t.repo || "") +
          "</td><td><code>" +
          esc(t.file || "") +
          "</code></td><td>" +
          esc(t.anchor_type || "") +
          "</td><td>" +
          esc(t.status || "") +
          "</td><td>" +
          esc(t.match_count || 0) +
          "</td><td>" +
          openBtn +
          "</td></tr>"
        );
      })
      .join("");

    return (
      '<section class="seam-grid">' +
      '<aside class="seam-list">' +
      list +
      '</aside>' +
      '<div class="seam-details">' +
      '<article class="card"><h3>' +
      esc(seam.title) +
      '</h3><p class="small"><code>' +
      esc(seam.id) +
      "</code>" +
      (seam.layer ? " · layer: " + esc(seam.layer) : "") +
      '</p><div class="decision-banner"><p><strong>Question:</strong> ' +
      esc(seam.decision_question || seam.question || "") +
      '</p><p><strong>Decision target:</strong> ' +
      esc(seam.decision_target || "") +
      '</p><p><strong>Answer:</strong> ' +
      esc(seam.decision_answer || "") +
      "</p></div>" +
      (seam.summary ? "<p>" + esc(seam.summary) + "</p>" : "") +
      (claimsHtml ? '<p><strong>Claims:</strong></p><ul class="mini-list">' + claimsHtml + "</ul>" : "") +
      '<div class="row">' +
      tags +
      "</div>" +
      (seam.why_it_matters
        ? '<p><strong>Why this matters:</strong> ' + esc(seam.why_it_matters) + "</p>"
        : "") +
      '<div class="seam-meta-grid">' +
      '<section class="card"><h4>Seam completeness (' +
      esc(completeness.passed || 0) +
      "/" +
      esc(completeness.total || 0) +
      ')</h4><table class="code-table"><tbody>' +
      (completenessRows || "<tr><td>n/a</td><td>No checklist items.</td></tr>") +
      "</tbody></table></section>" +
      recommendationHtml +
      "</div>" +
      '<section class="card"><h4>Compare panel</h4><table class="code-table"><tbody>' +
      (compareRows || "<tr><td colspan=\"3\">No comparison rows yet.</td></tr>") +
      "</tbody></table></section>" +
      '<div class="seam-meta-grid">' +
      '<section class="card"><h4>Reading checklist</h4>' +
      bulletList(seam.reading_checklist || [], "mini-list") +
      "</section>" +
      '<section class="card"><h4>Common pitfalls</h4>' +
      bulletList(seam.common_pitfalls || [], "mini-list") +
      "</section>" +
      "</div>" +
      (seam.related_seams && seam.related_seams.length
        ? '<p class="small"><strong>Related seams:</strong></p><div class="row">' +
          seamPills(seam.related_seams) +
          "</div>"
        : "") +
      "</article>" +
      '<section class="card"><h4>Story snippets (3-5)</h4><p class="small">Ordered proof chain: contract -> call site -> data/lifecycle -> divergence/example.</p></section>' +
      storyHtml +
      '<details class="evidence-drawer"><summary>Evidence drawer (all additional snippets, extraction plan, and quality)</summary>' +
      '<div class="card"><p class="small"><strong>Evidence plan:</strong> ordered targets define the argument chain for this seam.</p>' +
      '<div class="code-wrap"><table class="code-table"><tbody><tr><td><strong>#</strong></td><td><strong>target_id</strong></td><td><strong>story role</strong></td><td><strong>kind</strong></td><td><strong>repo</strong></td><td><strong>file</strong></td><td><strong>anchor</strong></td><td><strong>status</strong></td><td><strong>matches</strong></td><td><strong>view</strong></td></tr>' +
      (targetPlanRows || "<tr><td colspan=\"10\">No target plan.</td></tr>") +
      "</tbody></table></div>" +
      '<p class="small"><strong>Evidence quality:</strong> mean score ' +
      esc(meanScore) +
      " (1.0 strongest). Extractor mix below.</p>" +
      '<table class="code-table"><tbody>' +
      (extractorRows || "<tr><td>No extractor stats yet.</td></tr>") +
      "</tbody></table>" +
      '<div class="repo-tabs">' +
      tabBar +
      "</div></div>" +
      drawerHtml +
      "</details>" +
      "</div></section>"
    );
  }

  function searchView() {
    var q = state.searchQuery.trim().toLowerCase();
    var type = state.searchType;

    var results = [];
    if (!q) {
      return (
        '<article class="card"><h3>Search</h3><div class="row"><input class="search-input" id="search-input" placeholder="Search snippets, symbols, files, tags" value="" />' +
        '<select class="select" id="search-type"><option value="all">all</option><option value="snippets">snippets</option><option value="symbols">symbols</option><option value="grep">grep hits</option></select></div>' +
        '<p class="small">Type to search across snippet text, file paths, symbols, and tagged grep hits.</p></article>'
      );
    }

    function maybePush(itemType, payload, text) {
      if (text.indexOf(q) >= 0) {
        results.push({ type: itemType, payload: payload });
      }
    }

    if (type === "all" || type === "snippets") {
      model.snippets.forEach(function (sn) {
        var text = [
          sn.seam_title,
          sn.seam_id,
          sn.file_path,
          sn.code,
          (sn.tags || []).join(" "),
          sn.repo
        ]
          .join("\n")
          .toLowerCase();
        maybePush("snippet", sn, text);
      });
    }

    if (type === "all" || type === "symbols") {
      model.symbols.forEach(function (sym) {
        var text = [sym.repo, sym.file_path, sym.qualname, sym.name, sym.kind]
          .join("\n")
          .toLowerCase();
        maybePush("symbol", sym, text);
      });
    }

    if (type === "all" || type === "grep") {
      model.grepHits.forEach(function (hit) {
        var text = [hit.pattern, hit.repo, hit.file_path, hit.excerpt].join("\n").toLowerCase();
        maybePush("grep", hit, text);
      });
    }

    results = results.slice(0, 200);

    var rows = results
      .map(function (res) {
        if (res.type === "snippet") {
          var sn = res.payload;
          return (
            '<div class="result-item"><strong>Snippet:</strong> ' +
            esc(sn.seam_title) +
            ' · <code>' +
            esc(sn.file_path) +
            "</code> · L" +
            sn.start_line +
            "-" +
            sn.end_line +
            '<div class="row"><button class="btn" data-action="open-snippet" data-snippet="' +
            esc(sn.id) +
            '">Open in seam map</button><a class="btn" href="' +
            esc(sn.permalink) +
            '" target="_blank" rel="noreferrer">Permalink</a></div></div>'
          );
        }

        if (res.type === "symbol") {
          var sy = res.payload;
          var targetSnippet = findSnippetByFileLine(sy.repo, sy.file_path, sy.start_line);
          return (
            '<div class="result-item"><strong>Symbol:</strong> ' +
            esc(sy.qualname) +
            ' · <code>' +
            esc(sy.file_path) +
            "</code> · L" +
            sy.start_line +
            "-" +
            sy.end_line +
            (targetSnippet
              ? '<div class="row"><button class="btn" data-action="open-snippet" data-snippet="' +
                esc(targetSnippet.id) +
                '">Jump to nearest snippet</button></div>'
              : "") +
            "</div>"
          );
        }

        var gh = res.payload;
        var target = findSnippetByFileLine(gh.repo, gh.file_path, gh.line);
        return (
          '<div class="result-item"><strong>Grep hit:</strong> <code>' +
          esc(gh.pattern) +
          "</code> · " +
          esc(gh.repo) +
          " · <code>" +
          esc(gh.file_path) +
          "</code>:" +
          gh.line +
          (target
            ? '<div class="row"><button class="btn" data-action="open-snippet" data-snippet="' +
              esc(target.id) +
              '">Open related snippet</button></div>'
            : "") +
          "</div>"
        );
      })
      .join("");

    return (
      '<article class="card"><h3>Search</h3><div class="row"><input class="search-input" id="search-input" placeholder="Search snippets, symbols, files, tags" value="' +
      esc(state.searchQuery) +
      '" />' +
      '<select class="select" id="search-type"><option value="all"' +
      (state.searchType === "all" ? " selected" : "") +
      '>all</option><option value="snippets"' +
      (state.searchType === "snippets" ? " selected" : "") +
      '>snippets</option><option value="symbols"' +
      (state.searchType === "symbols" ? " selected" : "") +
      '>symbols</option><option value="grep"' +
      (state.searchType === "grep" ? " selected" : "") +
      '>grep hits</option></select></div><p class="small">Results: ' +
      results.length +
      " (showing max 200)</p>" +
      rows +
      "</article>"
    );
  }

  function diffView() {
    var snippetOptions = model.snippets
      .map(function (sn) {
        var label =
          sn.id +
          " | " +
          sn.repo +
          " | " +
          sn.file_path +
          ":" +
          sn.start_line +
          "-" +
          sn.end_line;
        return (
          '<option value="' +
          esc(sn.id) +
          '"' +
          (state.diffSnippetA === sn.id ? " selected" : "") +
          ">" +
          esc(label) +
          "</option>"
        );
      })
      .join("");

    var snippetOptionsB = model.snippets
      .map(function (sn) {
        var label =
          sn.id +
          " | " +
          sn.repo +
          " | " +
          sn.file_path +
          ":" +
          sn.start_line +
          "-" +
          sn.end_line;
        return (
          '<option value="' +
          esc(sn.id) +
          '"' +
          (state.diffSnippetB === sn.id ? " selected" : "") +
          ">" +
          esc(label) +
          "</option>"
        );
      })
      .join("");

    var files = model.files || [];
    var fileOptions = files
      .map(function (f) {
        var value = f.file_key;
        return (
          '<option value="' +
          esc(value) +
          '"' +
          (state.diffFileA === value ? " selected" : "") +
          ">" +
          esc(value) +
          "</option>"
        );
      })
      .join("");

    var fileOptionsB = files
      .map(function (f) {
        var value = f.file_key;
        return (
          '<option value="' +
          esc(value) +
          '"' +
          (state.diffFileB === value ? " selected" : "") +
          ">" +
          esc(value) +
          "</option>"
        );
      })
      .join("");

    var snippetDiffHtml = "<p class=\"small\">Choose two snippets to compare.</p>";
    if (state.diffSnippetA && state.diffSnippetB) {
      var sa = snippetById(state.diffSnippetA);
      var sb = snippetById(state.diffSnippetB);
      if (sa && sb) {
        snippetDiffHtml = renderDiff(sa.code, sb.code);
      }
    }

    var fileDiffHtml = "<p class=\"small\">Choose two files to compare.</p>";
    if (state.diffFileA && state.diffFileB) {
      var fa = model.filesByKey[state.diffFileA];
      var fb = model.filesByKey[state.diffFileB];
      if (fa && fb) {
        fileDiffHtml = renderDiff(fa.content, fb.content);
      }
    }

    var fileDiffSection = "";
    if (!model.filesLoaded) {
      fileDiffSection =
        '<section class="card"><h3>File-to-file diff</h3><p class="small">File index is lazy-loaded to keep seam pages fast.</p>' +
        '<button class="btn" data-action="load-files">Load file index</button></section>';
    } else {
      fileDiffSection =
        '<section class="card"><h3>File-to-file diff</h3>' +
        '<div class="row"><select class="select" id="diff-file-a"><option value="">Select file A</option>' +
        fileOptions +
        '</select><select class="select" id="diff-file-b"><option value="">Select file B</option>' +
        fileOptionsB +
        "</select></div>" +
        fileDiffHtml +
        "</section>";
    }

    return (
      '<div class="grid two">' +
      '<section class="card"><h3>Snippet-to-snippet diff</h3>' +
      '<div class="row"><select class="select" id="diff-snippet-a"><option value="">Select snippet A</option>' +
      snippetOptions +
      '</select><select class="select" id="diff-snippet-b"><option value="">Select snippet B</option>' +
      snippetOptionsB +
      "</select></div>" +
      snippetDiffHtml +
      '</section>' +
      fileDiffSection +
      "</div>"
    );
  }

  function renderDiff(aText, bText) {
    var lines = DiffUtil.lineDiff(aText, bText)
      .slice(0, 6000)
      .map(function (line) {
        var cls =
          line.type === "add" ? "diff-add" : line.type === "del" ? "diff-del" : "";
        var prefix = line.type === "add" ? "+ " : line.type === "del" ? "- " : "  ";
        return '<div class="diff-line ' + cls + '">' + esc(prefix + line.text) + "</div>";
      })
      .join("");
    return '<div class="diff-box">' + lines + "</div>";
  }

  function pocView() {
    var cards = (model.index.poc_plan || [])
      .map(function (poc) {
        var steps = (poc.steps || [])
          .map(function (step) {
            var seamLinks = (step.seams || [])
              .map(function (sid) {
                return (
                  '<button class="pill" data-action="open-seam" data-seam="' +
                  esc(sid) +
                  '">' +
                  esc(sid) +
                  "</button>"
                );
              })
              .join(" ");
            return "<li>" + esc(step.text) + '<div class="row">' + seamLinks + "</div></li>";
          })
          .join("");

        return (
          '<article class="card"><h3>' +
          esc(poc.title) +
          "</h3><ol>" +
          steps +
          "</ol></article>"
        );
      })
      .join("");

    return '<section class="grid">' + cards + "</section>";
  }

  function curationView() {
    var starred = model.snippets.filter(function (sn) {
      return state.starred.has(sn.id);
    });

    var body = "";
    if (!starred.length) {
      body = '<article class="card"><p>No starred snippets yet. Star snippets from Seam Map.</p></article>';
    } else {
      body = starred.map(snippetCard).join("");
    }

    return (
      '<section class="card"><h3>Curation mode</h3><p><strong>Starred snippets:</strong> ' +
      starred.length +
      '</p><p class="small">Use "Export Curated Markdown" in header to download selected snippets with permalinks.</p></section>' +
      body
    );
  }

  function reportView() {
    return '<article class="card"><h3>Embedded report.md</h3><pre class="report">' + esc(model.reportMd) + "</pre></article>";
  }

  function findSnippetByFileLine(repo, filePath, line) {
    for (var i = 0; i < model.snippets.length; i++) {
      var sn = model.snippets[i];
      if (sn.repo === repo && sn.file_path === filePath && line >= sn.start_line && line <= sn.end_line) {
        return sn;
      }
    }

    for (var j = 0; j < model.snippets.length; j++) {
      var sn2 = model.snippets[j];
      if (sn2.repo === repo && sn2.file_path === filePath) {
        return sn2;
      }
    }

    return null;
  }

  function render() {
    setActiveTab();

    if (!model.index) {
      app.innerHTML = '<article class="card"><p>Loading...</p></article>';
      return;
    }

    var html = "";
    if (state.view === "home") html = homeView();
    else if (state.view === "seams") html = seamMapView();
    else if (state.view === "search") html = searchView();
    else if (state.view === "diff") html = diffView();
    else if (state.view === "poc") html = pocView();
    else if (state.view === "curation") html = curationView();
    else if (state.view === "report") html = reportView();

    app.innerHTML = html;

    if (state.view === "search") {
      var searchInput = document.getElementById("search-input");
      var searchType = document.getElementById("search-type");
      if (searchInput) {
        searchInput.addEventListener("input", function (e) {
          state.searchQuery = e.target.value || "";
          render();
        });
      }
      if (searchType) {
        searchType.addEventListener("change", function (e) {
          state.searchType = e.target.value || "all";
          render();
        });
      }
    }

    if (state.view === "diff") {
      [
        ["diff-snippet-a", "diffSnippetA"],
        ["diff-snippet-b", "diffSnippetB"],
        ["diff-file-a", "diffFileA"],
        ["diff-file-b", "diffFileB"]
      ].forEach(function (pair) {
        var el = document.getElementById(pair[0]);
        if (el) {
          el.addEventListener("change", function (e) {
            state[pair[1]] = e.target.value || "";
            render();
          });
        }
      });
    }

    if (state.focusSnippetId) {
      var target = document.getElementById("snippet-" + state.focusSnippetId);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        target.style.outline = "2px solid #0f6f80";
        setTimeout(function () {
          target.style.outline = "";
        }, 1500);
      }
      state.focusSnippetId = "";
    }
  }

  function toggleStar(snippetId) {
    if (state.starred.has(snippetId)) state.starred.delete(snippetId);
    else state.starred.add(snippetId);
    persistStarred();
  }

  function exportCuratedMarkdown() {
    var selected = model.snippets.filter(function (sn) {
      return state.starred.has(sn.id);
    });
    var lines = [];
    lines.push("# Curated vLLM-Spyre Code Atlas Snippets");
    lines.push("");
    lines.push("Generated: " + new Date().toISOString());
    lines.push("");

    selected.forEach(function (sn) {
      lines.push("## " + sn.seam_title);
      lines.push("");
      lines.push("- Snippet ID: `" + sn.id + "`");
      lines.push("- Repo: `" + sn.repo + "`");
      lines.push("- File: `" + sn.file_path + "`:" + sn.start_line + "-" + sn.end_line);
      lines.push("- Commit: `" + sn.commit_short + "`");
      lines.push("- Permalink: " + sn.permalink);
      lines.push("");
      lines.push("```" + sn.language);
      lines.push(sn.code);
      lines.push("```");
      lines.push("");
    });

    var blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "curated-code-atlas-snippets.md";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function handleActions(e) {
    var t = e.target;
    var action = t.dataset.action;
    if (!action) return;

    if (action === "open-seam") {
      var seam = t.dataset.seam;
      if (seam) openSeam(seam);
      return;
    }

    if (action === "set-bucket") {
      state.seamBucket = t.dataset.bucket || "upstream-vllm";
      render();
      return;
    }

    if (action === "toggle-context") {
      var sid = t.dataset.snippet;
      state.showContext[sid] = !state.showContext[sid];
      render();
      return;
    }

    if (action === "toggle-file") {
      var sid2 = t.dataset.snippet;
      state.showFile[sid2] = !state.showFile[sid2];
      if (state.showFile[sid2] && !model.filesLoaded) {
        ensureFilesLoaded()
          .then(function () {
            render();
          })
          .catch(function (err) {
            console.error(err);
          });
      }
      render();
      return;
    }

    if (action === "toggle-snippet-size") {
      var sidSize = t.dataset.snippet;
      state.expandSnippet[sidSize] = !state.expandSnippet[sidSize];
      render();
      return;
    }

    if (action === "toggle-star") {
      var sid3 = t.dataset.snippet;
      toggleStar(sid3);
      render();
      return;
    }

    if (action === "copy-snippet") {
      var sid4 = t.dataset.snippet;
      var sn = snippetById(sid4);
      if (!sn) return;
      navigator.clipboard
        .writeText(sn.code)
        .then(function () {
          t.textContent = "Copied";
          setTimeout(function () {
            t.textContent = "Copy code";
          }, 900);
        })
        .catch(function () {
          t.textContent = "Copy failed";
        });
      return;
    }

    if (action === "open-snippet") {
      var sid5 = t.dataset.snippet;
      var sn2 = snippetById(sid5);
      if (sn2) {
        state.seamBucket = bucketForSnippet(sn2);
        openSeam(sn2.seam_id, sid5);
      }
      return;
    }

    if (action === "load-drawer") {
      var seamId = t.dataset.seam;
      if (seamId) {
        state.showDrawerContent[seamId] = true;
        render();
      }
      return;
    }

    if (action === "load-files") {
      ensureFilesLoaded()
        .then(function () {
          render();
        })
        .catch(function (err) {
          console.error(err);
        });
      return;
    }
  }

  function initNav() {
    navTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        state.view = tab.dataset.view;
        render();
      });
    });

    document.getElementById("export-curated").addEventListener("click", exportCuratedMarkdown);
    app.addEventListener("click", handleActions);
  }

  function loadData() {
    return Promise.all([
      fetch("./data/index.json").then(function (r) {
        return r.json();
      }),
      fetch("./data/snippets.json").then(function (r) {
        return r.json();
      }),
      fetch("./data/symbols.json").then(function (r) {
        return r.json();
      }),
      fetch("./data/grep_hits.json").then(function (r) {
        return r.json();
      }),
      fetch("./data/commits.json").then(function (r) {
        return r.json();
      }),
      fetch("./data/report.md").then(function (r) {
        return r.text();
      })
    ]).then(function (payload) {
      model.index = payload[0];
      model.snippets = payload[1];
      model.symbols = payload[2];
      model.grepHits = payload[3];
      model.commits = payload[4];
      model.reportMd = payload[5];
      model.snippetsById = {};
      model.snippets.forEach(function (sn) {
        model.snippetsById[sn.id] = sn;
      });

      if (!state.selectedSeamId && model.index.seams && model.index.seams.length) {
        state.selectedSeamId = model.index.seams[0].id;
      }

      var reportLink = document.getElementById("open-report");
      reportLink.href = "./data/report.md";
    });
  }

  loadStarred();
  initNav();
  loadData()
    .then(function () {
      render();
    })
    .catch(function (err) {
      app.innerHTML =
        '<article class="card"><h3>Failed to load atlas data</h3><pre class="report">' +
        esc(String(err && err.stack ? err.stack : err)) +
        "</pre></article>";
    });
})();
