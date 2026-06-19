# Per-sample explanations across all AVs (first 100 held-out samples)

For each held-out sample (same input across all models, matched by `idx`), the full explanation produced by each AV — the from-scratch base NLA, the control RL (λ=0), and each length-penalty model — with its reconstruction **FVE** (1 − mse/baseline; baseline mse_nrm = 0.6704). Source shows the **last 75 chars** of the document (the activation sits at its final token). Full data: [`-completions`](https://huggingface.co/datasets/syvb/nanonla-qwen3-8b-L24-completions).

### Sample 0
> …ients and families through the overwhelming process of Senior Care services

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 141 | 0.587 | Testimonial/review format signals continuation of positive patient praise about a local agency's services. The review follows a pattern of personal experience narrative with specific details.<br><br>Personal narrative momentum: the reviewer is mid-story, having established a problem (unanswered call) and resolution (finding The Right Path), now explaining why The Right Path suited them specifically.<br><br>Final phrase "families dealing with behavioral health services" ends mid-sentence, functioning as a noun phrase modifier — the next token must complete the main clause, likely describing what The Right Path did or what made it appropriate (e.g., "in my community" or "to meet our needs"). |
| control RL (λ=0) | 132 | 0.701 | Local news/feature article format: narrative profile of a community service initiative, establishing context for a personal story or testimony.<br><br>The phrase "impressed with her ability to make the complicated Medicaid services process understandable for families" sets up a quote or direct testimony from a family member or recipient, likely a parent describing how she helped them navigate healthcare.<br><br>The final token "services" ends a prepositional phrase ("for families seeking Medicaid services"), acting as the object of "for families," and the next tokens will likely continue with a period, comma, or transition phrase connecting to the specific individual story. |
| λ=0.001 | 96 | 0.647 | Local business/community announcement tone, praising a healthcare nonprofit's commitment to the community.<br><br>Narrative momentum: the quote is mid-explanation of how Angelos was inspired to support Family Hope, requiring completion of the motivation/situation.<br><br>Final token "services" closes the description of the problem Angelos faced; immediate expectation is continuation describing their response or transformation, likely "and" or a clause explaining their new path. |
| λ=0.002 | 72 | 0.560 | Narrative momentum: story building toward describing how the volunteer impacted the client's experience.<br><br>Direct quote continuation: the quote is mid-sentence, describing services received from the organization.<br><br>Final token "services" ends an incomplete thought mid-quote, expecting continuation explaining the emotional or educational impact of those services. |
| λ=0.006 | 32 | 0.613 | Narrative about program impact<br><br>explaining services provided<br><br>"services" closes sentence requiring continuation |
| λ=0.015 | 28 | 0.591 | community organization<br>support services<br>"educating families about behavioral health services" |
| λ=0.03 | 13 | 0.366 | and services |

### Sample 1
> …capes, reflecting the range of individual and collective experiences of the

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 127 | 0.480 | Narrative momentum from an exhibition catalog/description, establishing a thematic and visual analysis of the work "Buiten de stad," requiring continuation of critical commentary on the artwork's meaning.<br><br>Repetition/continuation of the phrase "outside the city" or the exhibition's name signals a recursive reference, likely reinforcing the theme being explored throughout the text.<br><br>Final token "the" opens a noun phrase mid-sentence, immediately requiring a noun or adjective — almost certainly "city" or a related urban/descriptive noun completing the thought about what "outside the city" entails. |
| control RL (λ=0) | 130 | 0.572 | Quotation continuation: the open quotation mark requires completion of the sentence about "living well in Amsterdam," which is mid-phrase.<br><br>Parallel structure established: "what people live with and what people live for" frames a broader participatory concept, suggesting the phrase "living well in Amsterdam" will complete a noun phrase referencing community or city dwellers.<br><br>Final token "the" is a determiner beginning a noun phrase inside the quoted sentence, immediately requiring a noun (e.g., "people," "city," "amsterdamers") to complete "living well in the [noun]." |
| λ=0.001 | 95 | 0.571 | Philosophical/reflective tone establishing conceptual depth about city living and identity formation.<br><br>The phrase "experiencing the" sets up a noun phrase completing the prepositional construction, requiring a specific concept like "city" or "metropolis."<br><br>Final token "the" is a definite article mid-noun-phrase, demanding an immediate noun or adjective+noun describing what people experience in Rotterdam's context. |
| λ=0.002 | 76 | 0.578 | Academic/institutional profile pattern describing research initiatives and collaboration outcomes.<br><br>The sentence is mid-thought, requiring completion of the research project description about residents of Rotterdam.<br><br>Final token "the" is a definite article beginning a noun phrase, almost certainly followed by "city," "area," or "city of Rotterdam." |
| λ=0.006 | 29 | 0.520 | research project description<br>plural noun reference required<br>"of the" needs completion |
| λ=0.015 | 27 | 0.517 | research project description<br>community perception survey<br>"perception of the" |
| λ=0.03 | 13 | 0.475 | of the |

### Sample 2
> …Officer, Vassilis Ragoussis. The funding round is led by Lansdowne Partners

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 110 | 0.785 | Business/venture capital announcement genre, establishing a funding event with company description, use of proceeds, and now investor details.<br><br>The text has shifted from company description to investor attribution, following the standard press release structure: announcement → deal terms → investor credit line.<br><br>Final token "Partners" completes the investor name "Lifeline Capital Partners," which in standard press releases is followed by a closing attribution such as "and" joining more investors, or a brief statement about the fund/portfolio. |
| control RL (λ=0) | 131 | 0.804 | Financial/funding announcement genre: company press release format signals continued disclosure of investor details and transaction specifics.<br><br>The sentence beginning "It was led by London-based Laurus Capital Partners" introduces the lead investor, requiring completion of the funding round details — amount, prior history, or additional investors expected next.<br><br>Final token "Partners" ends a proper noun phrase "Laurus Capital Partners" mid-sentence; immediate next tokens must complete the sentence, likely adding a relative clause or verb phrase describing what this firm did (e.g., "and closed with..." or "in a round that..."). |
| λ=0.001 | 80 | 0.725 | News article structure transitioning from narrative summary to formal deal announcement, now detailing fund sources.<br><br>"Investment from Laurus Capital" signals the start of attributing financial backers, with fund/round specifics expected next.<br><br>Final token "Laurus Partners" is mid-sentence after "from," requiring continuation with additional funding details or additional investor names. |
| λ=0.002 | 70 | 0.589 | Investment/financing announcement format establishes partnership details expected next.<br><br>The sentence "Backed by LSV Capital" signals investor attribution and deal structure details to follow.<br><br>"LSV Capital" is the final token, beginning a noun phrase requiring continuation of the institutional investor's involvement details. |
| λ=0.006 | 33 | 0.597 | funding announcement format<br>investment round details expected<br>"Led by Leumi Ventures" needs completion |
| λ=0.015 | 29 | 0.605 | investment deal announcement<br>venture capital details<br>"Led by Lendlay Capital" |
| λ=0.03 | 14 | 0.417 | Lincoln Partners |

### Sample 3
> …nt. EBay Inc. gave revenue and profit forecasts for the current period that

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 114 | 0.839 | Financial news reporting with analyst/revenue data structure: after stating Apple shares fell, specific financial guidance figures or comparisons are expected next.<br><br>The sentence "provided revenue and profit guidance that" sets up a clause requiring completion with specific forward-looking financial projections or performance expectations, likely involving specific dollar figures or percentage growth targets.<br><br>Final token "that" opens a relative clause modifying "guidance," requiring an immediate verb phrase describing what guidance stated or implied, e.g., "was above/below Wall Street's expectations." |
| control RL (λ=0) | 123 | 0.609 | Financial news report structure: earnings announcement follows standard format — revenue figures introduced, now expectations/outlook section expected.<br><br>Direct quote continuation pattern: the phrase "provided guidance that" is mid-sentence, requiring a noun clause describing investor expectations or forecasts, likely positive (e.g., "topped," "surpassed," "exceeded" estimates).<br><br>Final token "that" is a subordinating conjunction opening a relative clause, immediately requiring a verb phrase describing what the guidance stated, constrained by positive financial context ("surpassed estimates," "beat expectations"). |
| λ=0.001 | 79 | 0.632 | Financial news reporting establishing facts about earnings figures and revenue projections.<br><br>The sentence structure promises completion of what Amazon's guidance indicated, contrasting with investor expectations.<br><br>"that" is a relative pronoun ending an open clause, requiring an immediate verb phrase describing what the earnings guidance revealed, e.g., "that were higher/lower than expected." |
| λ=0.002 | 65 | 0.729 | Financial news article reporting Apple's earnings results and guidance.<br><br>Comparative contrast structure: "conservative guidance" sets up expected outcome versus market expectations.<br><br>The final clause "projections that" introduces a relative clause requiring a verb phrase describing guidance outcomes or analyst comparisons. |
| λ=0.006 | 31 | 0.677 | Financial earnings report context<br>comparison to analyst expectations<br>"results that" requires completing clause |
| λ=0.015 | 23 | 0.564 | earnings report<br>financial guidance<br>"that" |
| λ=0.03 | 13 | 0.255 | results that |

### Sample 4
> …For an extended effect, take up to 2 servings in a 24 hour period. Mix in a

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 133 | 0.773 | Product label format with standardized nutritional/usage instructions, following a strict formula supplement listing pattern (dosing, serving instructions, warnings).<br><br>The "Usage & Directions" section is mid-sentence, continuing a standard label formula instruction sequence that typically includes mixing/shaking directions and serving quantity (e.g., "Shake well in a shaker bottle... Use 30 mL...").<br><br>Final token "a" is an indefinite article beginning a noun phrase, requiring an immediate noun describing a container or mixing device — most likely "shaker bottle" or "shaker cup" based on supplement label conventions. |
| control RL (λ=0) | 120 | 0.724 | Product supplement instructions follow strict label/formulary conventions, requiring standardized directions language throughout. The text is mid-sentence in a preparation/mixing instruction.<br><br>The sentence structure "Mix in a [container]" signals a standard powder supplement mixing instruction, expecting specific equipment terminology like "shaker bottle," "blender," or "container with shaker."<br><br>Final token "a" is an indefinite article beginning a noun phrase, requiring a singular noun or adjective+noun immediately next, most likely a mixing container term consistent with bodybuilding supplement packaging. |
| λ=0.001 | 86 | 0.731 | Supplement product description format, with usage instructions following ingredient list conventions.<br><br>Shaking instruction in progress: "Mix in a" requires a container/shaker specification (e.g., "shake bottle" or "shaker cup").<br><br>Final token "a" is an indefinite article beginning a noun phrase specifying the mixing vessel, immediately constraining next token to a container noun. |
| λ=0.002 | 72 | 0.670 | Product supplement label format requires standardized serving instructions and mixing directions.<br><br>The sequence follows a nutritional supplement label structure, with usage instructions now expected.<br><br>"Mix in a" is an incomplete phrase requiring a container noun (e.g., "shaker bottle," "blender") to complete the mixing instruction. |
| λ=0.006 | 32 | 0.569 | Supplement product description<br><br>mixing instructions pattern<br><br>"mix in a" requires container type |
| λ=0.015 | 26 | 0.543 | fitness supplement instructions<br>preparation method<br>"Mix in a" |
| λ=0.03 | 14 | 0.126 | Mix in a |

### Sample 5
> …sustainability. DHL, UPS, FedEx, Amazon, Siemens, and Rivian are just a few

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 132 | 0.597 | List-continuation pattern: the text is enumerating companies leveraging AI (KPMG, Ernst & Young, McKinsey, Accenture, Capgemini, Deloitte, etc.) with "As a few" signaling an incomplete enumeration, strongly implying more names follow.<br><br>Argumentative momentum: the essay builds toward AI's business-value claims, supporting with examples and statistics throughout — next tokens will elaborate on specific AI implementations.<br><br>Final token "a few" is an incomplete quantifier phrase opening a specific list of named organizations, requiring continuation with a period or comma followed by additional company names. |
| control RL (λ=0) | 138 | 0.606 | Article follows a structured listicle format: introduction, benefits, challenges, now concluding with examples of AI adopters, signaling a list of company names is about to follow.<br><br>The narrative has built up to concrete real-world applications, and "a few" strongly implies a specific enumeration of companies (e.g., "of the companies" or "such as XYZ, ABC") is immediately expected next.<br><br>"a few" is a quantifier mid-phrase, part of an incomplete noun phrase ("a few [of the companies]" or "a few [like XYZ, ABC]"), requiring a noun or relative clause to complete it. |
| λ=0.001 | 80 | 0.613 | List/elaboration pattern: article introduces examples, building toward concrete company cases like Starbucks, Google, Microsoft.<br><br>Continuation of "a few" implies an incomplete enumeration of companies/organizations adopting AI sustainability tools.<br><br>Final token "a few" begins a list phrase requiring "of the companies" or similar complement specifying which organizations are examples. |
| λ=0.002 | 82 | 0.570 | Enumerated list structure with "A few" signaling continuation of company examples in the AI/ML adoption context.<br><br>Narrative momentum: article has introduced AI/ML benefits and is now providing concrete examples of adopters.<br><br>Final token "few" begins a quantified list of companies — next tokens must name specific organizations that have adopted AI/ML. |
| λ=0.006 | 37 | 0.746 | AI/tech article pattern<br><br>examples of AI adopters needed<br><br>"are just a few" requires completing the list |
| λ=0.015 | 22 | 0.499 | company examples<br>AI advancements<br>"a few" |
| λ=0.03 | 12 | 0.226 | A few |

### Sample 6
> …and more than a little nasty. There is NOTHING not to like about that guy. 

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 128 | 0.541 | Column-style sports journalism tone, blending analysis with humor ("homicidal maniacs," "stew in the cauldron") suggests continued witty commentary on NFL coaching hires.<br><br>The piece builds toward contrasting current coaching trends with past patterns, with each paragraph introducing a new angle or counterexample to support the "coaches getting younger" thesis.<br><br>Final sentence introduces a new coaching hire (Dennis Allen to replace Ken Whisenhunt in New Orleans) mid-sentence, with the period ending abruptly — the next tokens will continue this thought or pivot to another coach/news item. |
| control RL (λ=0) | 133 | 0.042 | Conversational sports journalism tone, blending opinion, stats, and commentary on NFL roster decisions, establishing informal but knowledgeable voice.<br><br>The article transitions from discussing 53-man roster cuts to "The Monday Take," suggesting a new weekly segment/column format — a recurring column header or opinion piece is beginning.<br><br>The final sentence ("And here's where you take the next 10 or so") appears to be a truncated or stylized column header/title, with "The Monday Take" likely followed by a question or call-to-action, constraining the next tokens to a question or prompt addressed to readers. |
| λ=0.001 | 85 | 0.641 | Sports news recap format with multiple topic transitions, suggesting another distinct sports observation follows.<br><br>Thematic momentum: each paragraph introduces a new football topic (Cowboys, Packers, Eagles, Steelers, Dolphins), implying continuation of similar observations.<br><br>Final period ends a Dolphins/NFL draft sentence; next token likely begins a new standalone sports-related observation or continuation of the current topic. |
| λ=0.002 | 64 | 0.720 | Column format with numbered/listed football observations, continuing a sports journalism piece.<br><br>Each paragraph delivers a standalone insight, maintaining conversational but analytical tone throughout.<br><br>The final sentence ends a thematic list item about NFL positions, suggesting another observation or closing remark follows next. |
| λ=0.006 | 26 | 0.637 | Sports column style<br>NFL draft analysis<br>New topic transition expected |
| λ=0.015 | 24 | 0.487 | sports analysis<br>NFL draft discussion<br>"Another thought..." |
| λ=0.03 | 11 | 0.445 |  |

### Sample 7
> …Best Management Practices (BMPs) were made as part of GZA’s work, including

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 116 | 0.769 | Bulleted list of best management practices introduced with "including," requiring specific examples to follow — e.g., "stormwater separators," "permeable pavement."<br><br>Formal environmental/government reporting tone throughout, with technical vocabulary like "stormwater management," "Municipal Separate Storm Sewer System" signals continued professional register.<br><br>Final token "including" is a list-introducing conjunction mid-sentence, demanding at least one or more specific BMP examples immediately after, consistent with the established pattern of concrete technical recommendations. |
| control RL (λ=0) | 124 | 0.740 | Bulleted list pattern established: "best management practices," "included" signals an upcoming list of specific BMPs (e.g., permeable pavement, bioswales, retention basins).<br><br>Technical/environmental domain: stormwater management project document with formal municipal planning register, requiring precise terminology like "permeable pavement," "vegetated swales," or "stormwater detention basins."<br><br>Final token "including" opens a list of specific BMP examples mid-sentence, grammatically requiring a comma-separated enumeration of stormwater control measures to follow immediately. |
| λ=0.001 | 88 | 0.744 | Technical/environmental report detailing facility upgrades at the USGS campus, following a structured problem-solution format.<br><br>The sentence is listing "best management practices," requiring specific examples of BMPs to complete the enumeration (e.g., stormwater detention, infiltration systems).<br><br>Final token "including" opens an incomplete list of specific best management practices, demanding continuation with concrete examples of such practices. |
| λ=0.002 | 54 | 0.748 | Ongoing list of BMP improvements being enumerated after "including."<br><br>Technical environmental/infrastructure reporting style with specific examples expected.<br><br>"including" is a list-introducing conjunction requiring concrete specific examples to follow immediately. |
| λ=0.006 | 31 | 0.704 | Wastewater infrastructure context<br>specific mitigation measures expected<br>"including" requires list continuation |
| λ=0.015 | 24 | 0.715 | stormwater BMPs<br>specific practices<br>"including" |
| λ=0.03 | 12 | 0.289 | including |

### Sample 8
> …Evans), the Hulk (Mark Ruffalo) and Hawkeye (Jeremy Renner). Joss Whedon is

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 132 | 0.733 | Film production announcement pattern: news brief describing a sequel release date, following a predictable format of cast, director, release details.<br><br>The text is mid-sentence introducing Joss Whedon's role as director, establishing a new subject after the release date was stated — "Ralph Fiennes will reprise his role as Dracula" introduces Whedon's contribution next.<br><br>Final token "is" opens a predicate clause about Joss Whedon's status or involvement, requiring a noun phrase or adjective describing his role, e.g., "is directing" or "is set to direct." |
| control RL (λ=0) | 124 | 0.787 | News article about a superhero film, establishing factual reportage tone with cast and plot details, guiding continued factual/celebrity reporting.<br><br>The text has introduced the director ("Joss Whedon") and the actors, creating a pattern of introducing key figures with their associated roles or contributions to the project.<br><br>"Joss Whedon is" — this is a predicate construction requiring completion, likely describing his role or status (e.g., "directing," "helming," "working on the film") in relation to the Marvel movie being discussed. |
| λ=0.001 | 85 | 0.751 | Production/director information pattern: article profiles Star Wars: The Force Awakens, establishing J.J. Abrams's role now.<br><br>Narrative momentum: sentence begun with "J.J. Abrams is" requires completing his current role or action regarding the film.<br><br>Final token "is" opens a predicate requiring immediate description of Abrams's present involvement or status on the project. |
| λ=0.002 | 72 | 0.760 | Ongoing news article about Marvel movie updates, establishing production details in sequence.<br><br>Joss Whedon's role in the film is being introduced, requiring his position/title next.<br><br>"Joss Whedon is" — a predicate beginning requires an immediate noun phrase describing his role on the film. |
| λ=0.006 | 27 | 0.577 | film production details<br><br>directorial credits<br><br>Ralph Fiennes is |
| λ=0.015 | 27 | 0.687 | film production<br>writer credit<br>"Joss Whedon is" |
| λ=0.03 | 14 | 0.211 | Joshua is |

### Sample 9
> … with a cargo bike, rider up to 120 kg and additional load of 70 kg! MIMO C

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 112 | 0.791 | Product description genre with structured marketing copy, introducing features and use cases in sequence.<br><br>The text has been building a case for a motorcycle with specific capabilities (e.g., obstacle avoidance, street riding), and a named product/feature "MOS C" is being introduced as the culmination.<br><br>"MOS C" is the start of a proper noun (likely "MOS CYCLE" or similar product name), requiring immediate completion of the brand/model name, followed by a capitalized noun phrase describing it. |
| control RL (λ=0) | 160 | — | _(extraction failed)_ |
| λ=0.001 | 88 | 0.526 | Product announcement format establishes brand name "MOS C" as focal point requiring completion.<br><br>The review/sales pitch momentum builds toward elaborating MOS C's features and value proposition.<br><br>Final token "C" is mid-product-name ("MOS C"), immediately requiring continuation with the remaining letters "YCLE" completing "MOS CYLE" followed by a description of its capabilities. |
| λ=0.002 | 79 | 0.765 | Product launch announcement building toward naming/branding details.<br><br>The text is completing a marketing description, with "MOS C" clearly beginning a product name.<br><br>Final token "MOS C" is the truncated brand name mid-sequence, strongly constraining next tokens to complete the model designation (e.g., "Cruiser"). |
| λ=0.006 | 28 | 0.647 | brand/product description format<br><br>model name continuation<br><br>MOS C[...] |
| λ=0.015 | 24 | 0.625 | product description<br>company name<br>"Mobility C" |
| λ=0.03 | 14 | 0.527 | MEC C |

### Sample 10
> …tatement. “We are thrilled to become strategic partners in the company that

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 124 | 0.695 | Direct quote from Steve Wynn is mid-sentence, requiring grammatical and thematic completion — likely praising his involvement or legacy at Cirque du Soleil.<br><br>The celebratory/biographical narrative arc frames Wynn as a figure whose creative contributions and personal life are being honored, setting expectations for positive, laudatory language.<br><br>Final token "that" opens a relative clause modifying "the company," immediately requiring a verb phrase describing what the company does, has done, or represents — e.g., "that has inspired me" or "that I founded." |
| control RL (λ=0) | 115 | 0.608 | Formal press release/news format, with structured company announcement and executive quote block following standard PR conventions.<br><br>The quote from John Malone is mid-sentence, praising Turner Broadcasting System — the narrative momentum demands completion of his gratitude statement, likely referencing specific achievements or Turner's role in entertainment.<br><br>Final token "that" opens a relative clause modifying "the company," requiring an immediate verb phrase continuing Malone's compliment, e.g., "that has given me so much success" or "that has brought me these great achievements." |
| λ=0.001 | 82 | 0.586 | Direct quote continuation: "The company that" requires completion describing Turner's legacy or Turner Broadcasting's role in the industry.<br><br>Celebratory tone and tribute structure: statement praising Turner's contributions and vision for the company's future.<br><br>Final token "that" opens a relative clause mid-sentence, immediately requiring a predicate describing Turner Broadcasting's qualities or achievements. |
| λ=0.002 | 76 | 0.549 | Continuation of a speech quote attributed to Jon Favreau about the Disney+ studio.<br><br>The sentence structure "the studio that" requires a relative clause describing the studio's significance to Favreau.<br><br>Final token "that" opens a defining relative clause, demanding an immediate verb phrase continuing the quote about the Disney+ studio. |
| λ=0.006 | 32 | 0.348 | celebrity quote structure<br>career/legacy framing<br>"that" relative clause requiring completion |
| λ=0.015 | 26 | 0.455 | celebrity quote<br>studio success<br>"the company that" |
| λ=0.03 | 14 | 0.306 | the company that |

### Sample 11
> …k of ability to navigate them by the rubble, although they had been capable

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 124 | 0.226 | Medical article explaining the coronavirus, establishing scientific rigor with detailed mechanisms of transmission. Formal informational tone maintained throughout.<br><br>Narrative momentum: the article has outlined transmission mechanisms (airborne/aerosol, contaminated surfaces) and is now expanding on surface transmission specifics, building toward completing a claim about the virus surviving on materials.<br><br>Final token "capable" ends mid-sentence describing what the virus was able to do — grammatically requires a verb complement immediately, most likely "of surviving" or "of spreading" on surfaces, continuing the established pattern of paired findings. |
| control RL (λ=0) | 119 | 0.083 | News article reporting on AI chatbot technology, establishing scientific findings about AI-generated content and user perceptions of authenticity.<br><br>Narrative momentum: the sentence is mid-clause describing what chatbots were able to do, specifically "able to" complete a capability, likely regarding generating factual or conversational responses.<br><br>Final token "able" is part of the phrase "able to," an incomplete infinitive construction requiring a verb (e.g., "able to generate," "able to distinguish") describing a specific capability chatbots demonstrated in the study. |
| λ=0.001 | 96 | 0.373 | Argumentative structure: article building case for why crypto should be regulated, with examples being enumerated.<br><br>Narrative momentum: "They can be capable" sets up a contrast explaining why crypto differs from traditional banks, implying limitations or capabilities.<br><br>Final token "capable" ends an incomplete clause "capable [of doing X]" requiring a specific verb phrase describing what crypto institutions can do, likely distinguishing them from traditional banks. |
| λ=0.002 | 78 | 0.421 | Narrative momentum: explaining how some doctors are tackling vaccine resistance despite limitations.<br><br>Conditional "even though" clause establishes contrast between resistance and what physicians are capable of.<br><br>Final token "capable" is mid-clause, requiring a verb phrase completing what doctors can do, e.g., "of offering" or "of providing". |
| λ=0.006 | 33 | -0.178 | Scientific reporting style<br>explanation of research methodology<br>"had the capability to" requires completion |
| λ=0.015 | 25 | 0.268 | vaccine effectiveness study<br>research findings<br>"were capable" |
| λ=0.03 | 14 | 0.086 | they had capable |

### Sample 12
> …nt. Why are you cast down, O my soul, and why are you in turmoil within me?

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 132 | -0.227 | Structured religious devotion content with consistent verse-by-verse commentary pattern, moving through Psalms sequentially, establishing strong genre expectations for continued Bible commentary.<br><br>The devotional commentary format pairs verse references with personal, reflective commentary, maintaining an accessible spiritual register throughout (e.g., "We all live this, don't we?" "We all have these moments").<br><br>The final line "When shall my heart and my soul be calm? When shall my soul find rest?" presents two parallel rhetorical questions requiring an answer; the pattern strongly predicts an immediately following response or elaborating commentary explaining how God answers such longing. |
| control RL (λ=0) | 138 | -0.159 | Sermonic/sermon content pattern: Bible verse, commentary, personal narrative, theological reflection — following a structured pastoral sermon format with citations.<br><br>Thematic focus on inner turmoil and trust in God: the passage addresses anxiety, conflict, and divine rest, requiring elaboration on spiritual peace as an antidote.<br><br>The final line is an incomplete scripture citation "(What shall I do, my soul, when troubled?)" from Psalm 42, mid-citation — the immediate next tokens must complete the reference, likely "(Psalm 42:11)" or continue with commentary explaining how trust in God resolves inner conflict. |
| λ=0.001 | 99 | 0.178 | Liturgical or devotional content pattern: Bible passages, hymns, and scripture references being listed sequentially.<br><br>Psalm 139 is the current scripture reference being cited, so verses or a question from that psalm is expected next.<br><br>Final token is "question?" — mid-sentence question from Psalm 139, demanding a continuation with the actual quoted verse or answer, likely "What is my soul's complaint?" |
| λ=0.002 | 84 | 0.204 | Biblical quote block is ongoing — a verse mid-completion requires continuation.<br><br>Psalm 130 is being cited, establishing the next line's scripture content.<br><br>The final token is an opening question mark mid-verse, demanding the corresponding biblical response to "Lord, why have you abandoned me?" — likely Psalm 130:1. |
| λ=0.006 | 28 | -0.152 | Prayer quote continuation<br>Biblical reference context<br>"What troubles me?" |
| λ=0.015 | 35 | -0.427 | Biblical passage<br>Psalm 130<br>"O my soul, what is my soul" |
| λ=0.03 | 15 | -0.430 | When am I discouraged? |

### Sample 13
> …confirms this – more than 60% of manufacturing workers are first generation

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 138 | 0.491 | Ongoing survey methodology description: report is detailing survey parameters (sample size, recruitment, response rate), building toward demographic breakdowns and response categories.<br><br>Narrative momentum: the sentence ending "with first generation" is mid-clause, likely continuing demographic segmentation (e.g., "immigrants" or "canadians") followed by further detail about response rates by population group.<br><br>Final token "generation" is the end of a noun phrase "first generation [immigrants]" or similar, syntactically requiring a noun to complete it — most likely "immigrants" — before the sentence continues with further response-rate demographic breakdowns. |
| control RL (λ=0) | 121 | 0.564 | Survey report presenting statistics about Toronto's ethnic/cultural diversity, maintaining informational and descriptive register throughout.<br><br>The text is mid-sentence completing a statistic about immigrant populations, listing categories of non-status/immigrant residents with "first generation" as the first descriptor, implying further qualifying demographic details follow.<br><br>"first generation" is the opening of a noun phrase describing a subgroup of non-status residents; the next token must complete this term, almost certainly with "Canadian" (i.e., "first generation Canadian immigrants"), followed by additional demographic subcategories. |
| λ=0.001 | 93 | 0.418 | Academic report style with formal policy implications section building evidence for language/immigration policy recommendations.<br><br>Continuation of demographic breakdown: the paragraph contrasts Toronto-born English speakers with immigrant English speakers, establishing pattern requiring completion of statistic.<br><br>Final token "generation" is mid-phrase in a descriptor clause ("one-generation [immigrants]"), requiring immediate completion such as "immigrants" followed by further demographic statistics. |
| λ=0.002 | 75 | 0.628 | Report findings being enumerated: statistics and demographic analysis about Scarborough's youth population.<br><br>Continuation of a list of Scarborough demographic data points about youth and immigration.<br><br>Final token "generation" begins a noun phrase describing Canadian immigration status; next tokens will complete "first-generation" (immigrants) likely followed by further statistics. |
| λ=0.006 | 30 | 0.417 | Survey methodology details<br><br>demographic breakdown in progress<br><br>"first generation" requires completion |
| λ=0.015 | 25 | 0.376 | demographic breakdown<br>immigration statistics<br>"first generation" |
| λ=0.03 | 13 | -0.147 | first generation |

### Sample 14
> … an account? Email us at firstname.lastname@example.org and we can help you

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 100 | 0.570 | Formal government/municipal communications style maintained throughout, with institutional register for public service announcements.<br><br>Narrative momentum building toward a helpful resolution: after establishing the issue of missing library access and offering to help, a solution or contact instruction is expected.<br><br>Final fragment "We'll help you" is an incomplete transitive clause mid-thought, immediately requiring a direct object or adverb phrase specifying what assistance will be provided or how to proceed. |
| control RL (λ=0) | 115 | 0.590 | Customer service/technical help desk register: formal yet reassuring tone advising users through a problem resolution process.<br><br>Narrative momentum toward resolving a login/user account issue: the user is being guided to fix an inability to access their account, with a clear call-to-action for customer support to assist.<br><br>Final token "you" is part of an incomplete sentence beginning "Help us help you," which expects a verb phrase continuation describing what customer support will do — likely "find," "resolve," or "access" the account. |
| λ=0.001 | 76 | 0.601 | Customer support response mid-sentence, following a problem-reporting email with resolution steps being offered.<br><br>The sentence "will help you" sets up a directive completing the action of assisting the customer.<br><br>Final token "you" ends an incomplete clause requiring a verb phrase continuation like "resolve this" or "find your account." |
| λ=0.002 | 65 | 0.698 | Customer service response guiding account access issue resolution, maintaining supportive tone.<br><br>The system is offering assistance with account setup or access after a purchase.<br><br>Final phrase "help you" strongly anticipates a verb completion like "set up your account" or "access your service." |
| λ=0.006 | 31 | 0.447 | Customer service reply context<br>instructional/helpful tone<br>"help you" requires completion |
| λ=0.015 | 24 | 0.428 | library help request<br>customer service<br>"help you" |
| λ=0.03 | 13 | 0.305 | help you |

### Sample 15
> … bring a little whimsy to your life as well.” – Lacey Bows are by Laco Hand

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 118 | 0.678 | Product listing format for handmade jewelry, describing materials/origin of each item in a numbered list pattern (items 1–4 so far).<br><br>The text follows a consistent structure: product name, origin, materials, then detailed description — establishing expectations for completing item 4's description.<br><br>"Holli's Hand" is a proper noun mid-construction, almost certainly the beginning of "Holli's Handmade" or "Holli's Handcrafted," a brand name requiring immediate completion of this hyphenated word/phrase. |
| control RL (λ=0) | 120 | 0.437 | Product description format for a children's jewelry brand, establishing style and origin — next content continues brand history/heritage.<br><br>Brand name "Hand" is being completed — "Karina Hand" likely refers to founder Karina Hand (a real person), so her last name continuation is strongly expected.<br><br>"Karina Hand" is an incomplete proper noun mid-sentence; the immediate next token must be her surname completion, likely followed by her role/title or a continuation of the brand's origin story ("Karina Hand founded..." etc.). |
| λ=0.001 | 98 | 0.716 | Product description listing brand and supplier details, establishing e-commerce retail register.<br><br>The sentence is mid-completion naming a supplier/partner brand "Lilah Hand [something]" requiring the product's full brand name.<br><br>"Hand" is the final token, the start of a proper noun brand name (e.g., "Handmade," "Hand Painted," or a combined brand name), requiring immediate continuation of that brand name. |
| λ=0.002 | 64 | 0.694 | Product listing description building toward brand/manufacturer attribution.<br><br>List of curated handmade products being enumerated with origin details.<br><br>"Katie Hams Hand" is mid-sentence, strongly expecting a continuation like "Katie Hams Handmade" or similar brand name. |
| λ=0.006 | 36 | 0.567 | product description narrative<br><br>company/brand attribution pattern<br><br>"KA Hand" is the start of "KA Handmade" |
| λ=0.015 | 24 | -0.070 | product description<br>brand name<br>"Kris Hand" |
| λ=0.03 | 12 | 0.220 | Hand |

### Sample 16
> … on it to finish the level. The most different objectives guarantee fun and

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 119 | 0.773 | Promotional/informational app store listing pattern: repetitive "Download" call-to-action signals a template-style description block.<br><br>Game description narrative momentum: after listing features ("different animals, interesting situations"), a positive outcome or benefit is expected, completing the "fun and [adjective/noun]" phrase.<br><br>Final token "and" is a coordinating conjunction mid-phrase, requiring a parallel adjective or noun (e.g., "challenge," "excitement," "rewards") to complete the compound descriptor in the game's description. |
| control RL (λ=0) | 122 | 0.694 | Game description genre following a structured promotional/entertainment format, with gameplay mechanics, features, and appeal already enumerated.<br><br>Narrative momentum toward completing the sentence about gameplay characteristics — "fun and" sets up a paired adjective or noun phrase (e.g., "fun and addictive," "fun and challenging") completing the player experience pitch.<br><br>Final token "and" is a coordinating conjunction mid-phrase, requiring an immediate continuation — almost certainly another adjective or noun completing the pair describing what makes the game enjoyable ("fun and challenging," "fun and rewarding"). |
| λ=0.001 | 81 | 0.724 | Game description/marketing copy pattern establishes continuation of feature-highlighting language.<br><br>The phrase "fun and" sets up a paired conjunction requiring a complementary noun phrase (e.g., "challenge," "entertainment," "excitement").<br><br>Final token "and" is a coordinating conjunction mid-phrase, demanding an immediate continuation completing the fun description. |
| λ=0.002 | 62 | 0.711 | Promotional description maintaining casual, enthusiastic tone throughout.<br><br>The sentence promises "fun and [something else]" continuing the list of gameplay qualities.<br><br>Final token "and" is a coordinating conjunction mid-list, requiring another adjective or noun completing "fun and..." |
| λ=0.006 | 33 | 0.646 | game description format<br><br>fun and entertainment promised<br><br>"and" requires a noun phrase completing the list |
| λ=0.015 | 23 | 0.685 | game description<br>fun elements<br>"fun and" |
| λ=0.03 | 13 | 0.512 | fun and |

### Sample 17
> …n ground we share as Californians: We all want healthier oceans, and we all

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 103 | 0.471 | Ongoing direct quote from Oregon governor Brown establishes a policy speech pattern, requiring continuation of formal political rhetoric.<br><br>The speech emphasizes balancing growth and preservation, suggesting next text will elaborate on shared Oregon values and future planning.<br><br>Final token "all" ends mid-sentence within a subordinate clause ("all of us all"), requiring a verb or adjective continuation like "want to protect" or "believe in," completing the shared-values argument central to the policy vision. |
| control RL (λ=0) | 122 | 0.526 | Ongoing direct quotation from Oregon Gov. John Kitzhaber requires continuation of his speech's formal policy rhetoric.<br><br>The phrase "We all" opens a shared collective noun construction, typical of political speeches emphasizing bipartisan or universal agreement, e.g., "we all value/need/want" something related to coastal/ocean sustainability.<br><br>Final token "all" is mid-clause within a direct quote starting "We all," syntactically requiring a verb phrase next, most likely expressing shared values, benefits, or commitments regarding Oregon's coastal resources. |
| λ=0.001 | 109 | 0.343 | Ongoing argumentative structure: speech addresses coastal resource management, building toward shared community values.<br><br>Repetition/continuation pattern: "All of us and all of us" structure implies a parallel list of collective stakeholder values, e.g., "all of us care about the coast" or similar communal concern.<br><br>Final token "all" is the start of a repeated phrase mid-sentence, immediately requiring continuation like "we all care about" or "all of us depend on." |
| λ=0.002 | 66 | 0.579 | Ongoing direct quote from Rep. Greg Walden requires continuation in his voice.<br><br>Environmental advocacy speech building toward shared stakeholder concerns about the Klickitat Valley.<br><br>"We all" is mid-sentence, syntactically requiring a verb phrase expressing collective interest or benefit. |
| λ=0.006 | 24 | 0.507 | policy speech structure<br>shared stake theme<br><br>we all |
| λ=0.015 | 23 | 0.496 | political debate<br>community values<br>"we all" |
| λ=0.03 | 13 | 0.330 | we all |

### Sample 18
> … is part of the "Books on Canvas" Series. It is an interpretation of Edmund

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 133 | 0.447 | Book review genre following a structured evaluative framework (plot, characters, prose, recommendation), now moving to author bio/credits section. The pattern "This is a review... of..." signals formal review structure, with the author bio now introduced.<br><br>The sentence beginning "This is a review" transitions into an author bio, requiring completion of the biography sentence — specifically Edmund D. Moreau's name and description.<br><br>Final token "Edmund" is part of a bio introduction mid-sentence, immediately requiring his full surname ("D. Moreau") and subsequent biographical description of the author's background. |
| control RL (λ=0) | 116 | 0.282 | Structured event listing format establishes context of describing an author's work/visiting.<br><br>The sentence introduces "Edmund" mid-name, requiring completion of the full name "Edmund D. Moreau" or similar, followed by an explanation of his work/relevance to the event.<br><br>Final token "Edmund" is the first name of an author being introduced, part of a sentence about reading from his works; next tokens must complete his full name and likely a relative clause explaining his writings or connection to the theme. |
| λ=0.001 | 96 | 0.328 | Book/bibliography description genre with event promotion context, establishing a reader's guide structure.<br><br>The phrase "explore Edmund" promises a specific author introduction, requiring the full name and likely a descriptor or book reference to follow.<br><br>Final token "Edmund" is the beginning of an author's full name mid-sentence, immediately requiring completion of the surname ("Dekker" most likely) followed by book context. |
| λ=0.002 | 90 | 0.489 | Book review/listing pattern established, requiring completion of bibliographic information about the author.<br><br>The review is building to describe Edmund De Waal's biography and work in relation to "The Hare with Amber Eyes."<br><br>Final token "Edmund" is the start of Edmund De Waal's full name, requiring immediate completion with "De Waal" and likely a brief biographical note. |
| λ=0.006 | 30 | 0.356 | Book review narrative<br><br>biographical introduction pattern<br><br>" Edmund" requires full name completion |
| λ=0.015 | 23 | 0.361 | book review<br>Edmund<br>"Edmund" |
| λ=0.03 | 13 | 0.364 | about Edmund |

### Sample 19
> …irectly from Youtube, they may or may not be most relevant to the word "Lom

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 122 | -0.079 | Blog/article aggregation format with labeled sections ("Latest," "Related Articles," "Most Viewed") repeating throughout, establishing a clear template for content listing.<br><br>List continuation pattern: "Most Viewed" section is mid-sequence, with multiple articles listed in descending order by views, requiring more entries to complete the series.<br><br>Final token "Lom" is a partial word, almost certainly "Lomoc" (Lomocash), the recurring article title from the first post, requiring completion of that word before the truncated article title can continue. |
| control RL (λ=0) | 137 | 0.764 | Acronym completion: "Lom" is the beginning of "LoMo," the well-known loanword from the Japanese mobile lending service "Lending Loop."<br><br>The phrase "Lom-" signals a named brand/product introduction, requiring the full brand name ("LoMo" or "LoMo Money") followed by a description of how mobile lending works in India.<br><br>Final token "Lom" is a partial word mid-noun-phrase, immediately constraining the next tokens to complete "LoMo" (likely "LoMo Money" or "LoMo Mobile") before describing its functionality in the Indian mobile-first market. |
| λ=0.001 | 122 | -0.264 | Financial journalism establishing a loan agreement between Lomé's city council and IFC, requiring continuation of the deal's terms and parties.<br><br>Narrative momentum: the article has concluded the deal's summary and is now transitioning to an expert analysis or quote, signaled by "According to Lomé-based microfinance expert."<br><br>Final token "Lom" is the truncated start of "Lomé" within a new quote attribution, immediately requiring completion as "é-based microfinance expert" or "é's [name]" to introduce an expert statement. |
| λ=0.002 | 89 | 0.464 | Narrative shift to a new paragraph with "Last week, Lom" signaling a follow-up story about Lom.<br><br>Chronological storytelling pattern continues, building on the Lom loan program's expansion.<br><br>Final token "Lom" is mid-word or mid-phrase, beginning the proper noun "Lomahafo" or "Lom," requiring immediate completion of that name. |
| λ=0.006 | 34 | 0.674 | Financial case study narrative<br><br>Loan repayment process explanation<br><br>"Lom" is the start of "Lom" |
| λ=0.015 | 22 | 0.681 | loan description<br>Lom<br>"Lom" |
| λ=0.03 | 12 | 0.491 | Lom |

### Sample 20
> …y, VIP members will receive double VIP points between March 25th and April 

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 133 | -0.354 | Numerical completion in progress: the text is mid-sentence describing the 40-point reward, requiring the full redemption threshold (e.g., "40 points to redeem").<br><br>Promotional/retail marketing copy style: formal brand messaging with specific figures ("80 points," "15000 points," "21000 points") drives continuation with similar concrete rewards or redemption terms.<br><br>Final token "to" is a preposition completing "40 points to," immediately requiring a verb or noun phrase describing what happens next, most likely "redeem" or "earn." |
| control RL (λ=0) | 126 | -0.479 | Structured list pattern: each entry has a price, percentage, and reward description, suggesting another entry or completion is expected.<br><br>The sentence beginning "As a member of..." is mid-clause, establishing a conditional/reward context typical of promotional content (e.g., "As a member of the loyalty program, you'll receive...").<br><br>Final token "2012-" is a truncated date within "from 1 September 2012-" indicating the next token completes the year range, almost certainly "2013" to close the time-frame. |
| λ=0.001 | 88 | 0.074 | Promotional announcement pattern: details being listed (points, program name) continue with membership benefits.<br><br>Incomplete sentence structure: "only when you are a member of" requires completing the named loyalty program membership level.<br><br>Final token "a" begins an indefinite article mid-phrase, directly constraining the next token to a membership tier noun like "premium member" or "gold member." |
| λ=0.002 | 95 | -0.283 | Promotional/marketing copy for British Airways ClubWorld program, maintaining persuasive tone throughout.<br><br>Continuation of a structured benefit listing, with each point building on membership advantages.<br><br>The final token "l" is the beginning of "l" in "l" — completing "becomes a full member" or "l" as part of "l" continues "becomes a full member or l [login/register]" |
| λ=0.006 | 32 | -0.021 | loyalty program benefits explained<br>yearly points threshold pattern<br>"to" begins completion phrase |
| λ=0.015 | 28 | -0.075 | loyalty program details<br>membership tier<br>"becomes a" incomplete |
| λ=0.03 | 16 | -0.262 | the "m" or |

### Sample 21
> …nes, some classical masterpieces by Liszt and Chopin along with the biggest

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 159 | 0.700 | Celebrity biography/promotional press release pattern establishes formal description tone throughout. The text follows a structured profile format introducing credentials and background.<br><br>Narrative momentum: a list of career achievements is being enumerated; the phrase "from the most iconic pop hits to the biggest" promises a contrasting musical category (e.g., "hits" completes the contrast, implying something broader like "rock hits" or "world music").<br><br>Final token "biggest" is the second part of a parallel construction ("the most iconic pop hits to the biggest"), syntactically demanding a noun phrase or adjective to complete the contrast, almost certainly referencing another genre or scale (e.g., "biggest rock hits" or "biggest world music chart-toppers"). |
| control RL (λ=0) | 131 | 0.709 | Promotional/introductory music artist bio pattern: enthusiastic description of a new album release, building toward features and highlights.<br><br>The phrase "mixing some of the biggest" strongly implies a contrast or continuation — likely "hits" or "hit songs" to complete the well-known "mixing the biggest hits" cliché in music journalism.<br><br>Final token "biggest" is mid-noun-phrase, an adjective requiring an immediate noun complement (e.g., "hits," "songs," "moments") to complete the collocation, likely followed by additional album description or release details. |
| λ=0.001 | 96 | 0.711 | Promotional/press release genre establishing artist credentials and tour context throughout.<br><br>The phrase "combine the biggest" sets up a list of musical styles/genres, requiring continuation naming specific elements (e.g., "hit songs," "international hits").<br><br>Final token "biggest" is mid-phrase, part of an infinitive construction "combine the biggest X," demanding an immediate noun or adjective complement to complete the thought. |
| λ=0.002 | 75 | 0.725 | Promotional press release tone continues, describing the artist's live performances.<br><br>List of performance themes being enumerated: "classic, the most popular, and the biggest" signals an ongoing series.<br><br>"the biggest" is the final fragment, requiring a noun phrase describing hit songs or hits, e.g., "hits." |
| λ=0.006 | 28 | 0.616 | band description continuation<br><br>music genre focus<br><br>"the biggest" requires completion |
| λ=0.015 | 24 | 0.618 | musician biography<br>career highlights<br>"the biggest" |
| λ=0.03 | 13 | 0.374 | the biggest |

### Sample 22
> …chieved through visual merchandising. The product features and benefits are

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 110 | 0.754 | Marketing/product description genre with promotional language ("excellent," "unique," "visually appealing") driving continuation of feature elaboration.<br><br>List-like enumeration of brand features: product description structure promises to continue detailing visual, functional, and sensory attributes of the packaging.<br><br>Final token "are" begins a predicate completing "product features are," requiring immediate continuation of a verb phrase describing what features do — e.g., "highlighted," "communicated," or "made visible" through design elements. |
| control RL (λ=0) | 134 | 0.830 | Marketing case study format (product launch) establishes a pattern of describing brand strategy, objectives, and execution in sequence.<br><br>The sentence structure "The product features and benefits are" sets up a completion clause describing what these features/benefits *do* or *are communicated/expressed*, expecting a verb phrase like "communicated through," "highlighted in," or "expressed visually."<br><br>Final token "are" is a linking verb beginning a predicate, requiring an adjective, noun phrase, or verb phrase immediately next (e.g., "clearly communicated," "visually expressed," "highlighted"). |
| λ=0.001 | 84 | 0.779 | Marketing copy describing a branded experience, requiring continuation of feature elaboration about product benefits.<br><br>The phrase "product features are" sets up a predicate completing what features are doing — likely explaining their communication/impact.<br><br>Final token "are" opens a predicate clause mid-sentence, immediately requiring a verb complement describing how product features are presented, conveyed, or communicated. |
| λ=0.002 | 68 | 0.619 | List/continuation pattern: the sentence is building toward completing a description of product features.<br><br>Brand identity/identity language momentum: promotional retail description consistently highlights sensory and emotional qualities.<br><br>Final token "are" opens a predicate requiring completion describing how product features are presented/communicated. |
| λ=0.006 | 30 | 0.559 | Product design context<br><br>value propositions being listed<br><br>"product features are" requires completion |
| λ=0.015 | 25 | 0.586 | product display concepts<br>feature presentation<br>"product features are" |
| λ=0.03 | 15 | 0.302 | the product features are |

### Sample 23
> …o changes: "SELECT COUNT h_sm_relationship_entities.h_entity_id FROM h_itsm

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 156 | -0.044 | SQL query result pattern: structured query showing a joined table `ims_ticket_446556874180010806` with fields like `ims_workflow`, `ims_issuetypes`, `ims_ticket` — next tokens likely continue this table or output.<br><br>List continuation: "ims_ticket" appears multiple times in column names, suggesting a pattern of `ims_prefix` repeated, possibly completing another column reference.<br><br>Final token `ims_issuetypes` is a partial SQL column alias, part of a joined table definition; immediate next tokens are likely another comma-separated column, table name, or query continuation like `as` or a `from` clause. |
| control RL (λ=0) | 121 | 0.199 | SQL query syntax in progress: a SELECT statement is being constructed, requiring the completion of the table name and field references.<br><br>Narrative context of data migration problem (restructure CMDB DB) drives expectation of technical field/column names like `ticket_id`, `incident`, or `description`.<br><br>The final token `isam` is the beginning of a table name (`isam.ticket` or `isam.ticketid`), and the immediate next tokens must complete this table identifier, likely a period followed by a field name or another table reference. |
| λ=0.001 | 99 | 0.311 | Technical documentation explaining an API/field mapping issue, establishing context for a specific solution.<br><br>The narrative is mid-sentence describing a table structure involving "ITSM" and "itilitsm," suggesting a table column name pattern continues.<br><br>Final token "itilitsm" is part of a table reference "itilitsm.ticket" — immediate expectation is continuation of field/table names or a description of what follows the table. |
| λ=0.002 | 73 | 0.269 | Database/table structure being described, expecting column/table names and schema details.<br><br>Narrative momentum: the response is cataloguing database entities in an informational/help register.<br><br>Final token "itms" is part of the truncated table name "itms_incident," immediately requiring continuation with the remaining table/column name. |
| λ=0.006 | 35 | 0.491 | technical support context<br><br>database table structure explanation<br><br>"from ism_itsm" opens a table name |
| λ=0.015 | 26 | 0.192 | ticket system<br>technical database<br>"itil__itsm" |
| λ=0.03 | 14 | -0.026 | ita_itam |

### Sample 24
> …ealth assistance to be in that context and especially important considering

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 122 | 0.418 | Political policy advocacy speech pattern: formal legislative/proclamation language urging action with rhetorical devices and repeated emphasis on urgency ("urgent need," "imperative").<br><br>Narrative momentum toward justification: the "considering" clause opens a causal or conditional argument, requiring explanation of why vaccination is more urgent now than earlier.<br><br>Final token "considering" is a subordinating conjunction beginning a clause that must immediately provide a specific context, reason, or statistic (e.g., demographic data, disease trends, or current crisis), consistent with the urgent tone established. |
| control RL (λ=0) | 129 | 0.665 | Formal advocacy/legislative speech genre, using official register throughout — expects continued policy rationale and evidence-building.<br><br>The sentence ending with "considering" introduces a subordinate clause that must complete with specific factors explaining why health services are urgently needed (e.g., population health statistics, pandemic impact, resource scarcity).<br><br>The final token "considering" is a subordinating conjunction opening a reason clause, requiring an immediate noun phrase or gerund describing the urgent public health context (e.g., "the pandemic," "limited resources," or "current health challenges in Vanuatu"). |
| λ=0.001 | 83 | 0.662 | Argumentative momentum: speech building case for urgent malaria vaccine procurement, with reasoning being completed.<br><br>Causal clause continuation: "considering" introduces a specific reason/elaboration about the malaria threat's severity in the region.<br><br>Final token "considering" opens a subordinate clause requiring an explanation of the epidemiological or humanitarian context justifying the vaccine priority. |
| λ=0.002 | 72 | 0.589 | Argumentative momentum: explaining why dental care is urgent for Tongan people.<br><br>Causal explanation expected: "considering" introduces supporting evidence for the problem's urgency.<br><br>Final token "considering" opens a subordinate clause requiring specific contextual reasons (e.g., pandemic, workforce challenges, disease burden). |
| λ=0.006 | 32 | 0.425 | health advocacy momentum<br><br>covid-19 context required<br>"considering" demands reason |
| λ=0.015 | 24 | 0.301 | public health concern<br>government commitment<br>"considering" |
| λ=0.03 | 13 | 0.359 | considering |

### Sample 25
> … thus can be utilized for various gas sensing applications. Publishing Date

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 139 | 0.575 | Scientific abstract metadata format — fields like Journal, Volume, Authors, Abstract, Keywords, Pages follow structured academic conventions.<br><br>Bibliographic record listing expected: "Publishing Date" field requires a specific date (e.g., "2021", "2021-12-01") in a standardized date format, continuing the metadata table structure.<br><br>"Publishing Date" is the final metadata field label, directly requiring its corresponding value — a year, date, or month-year (e.g., "2021", "December 2021") to complete the bibliographic entry. |
| control RL (λ=0) | 120 | 0.360 | Academic paper metadata format, following structured journal article citation template with author list, abstract, keywords, journal details.<br><br>The "Publication Date" field signals a chronological record entry, establishing temporal context; a specific date string (day/month/year) is strongly expected next.<br><br>"Publication Date" is a metadata field label acting as the start of a structured citation row; it directly constrains the next token to be a colon, date string (e.g., ": 2021"), or similar date value completing the publication information entry. |
| λ=0.001 | 98 | 0.413 | Academic paper metadata table structure, with fields like "Abstract," "Citations," "Downloads," "Download Pdf."<br><br>Publication metadata pattern: after "Published Date," a specific date (e.g., "2023-08-15") or date format is expected next.<br><br>"Published Date" is the last field label in a metadata table row, directly requiring a colon or date value to follow immediately. |
| λ=0.002 | 72 | 0.281 | Academic journal citation/abstract format establishes structured metadata fields.<br><br>Publication date section is a standard bibliographic field requiring a date value.<br><br>"Publication Date" is a label/header, directly expecting a colon and a date (e.g., ": 2024-03-25"). |
| λ=0.006 | 35 | 0.383 | Academic paper metadata format<br><br>Publication date field follows<br><br>"Publication Date" is a label requiring a date |
| λ=0.015 | 23 | 0.354 | academic paper<br>publication metadata<br>"Publication Date" |
| λ=0.03 | 13 | -0.369 | Publication Date |

### Sample 26
> …ironment with mass spectrometry have damaged the proteins and their ability

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 134 | 0.639 | Scientific explanation momentum: the paragraph is detailing a technical finding about lipid removal from DNA samples, building toward a specific consequence or mechanism.<br><br>Parallel structure pattern: the sentence describes two simultaneous effects ("lipids are removed and their ability") implying a paired list of what is lost alongside what is removed — "ability to hybridize to DNA" or similar function is strongly expected next.<br><br>Final token "ability" is mid-phrase, part of "their ability to [verb]," requiring an infinitive verb completing what the removed lipids can no longer do, likely "to hybridize" or "to bind." |
| control RL (λ=0) | 117 | 0.696 | Scientific article explaining protein engineering technique, building toward a technical explanation of why purified proteins "lose their ability."<br><br>The sentence structure "lose their ability" demands a complement — a verb phrase describing what proteins can no longer do (e.g., "fold properly," "function," "maintain structure").<br><br>Final token "ability" is a noun mid-complement, part of "lose their ability [to do X]" — the next token must be "to" followed by a verb describing functional impairment of the purified proteins. |
| λ=0.001 | 107 | 0.466 | Scientific explanation building toward a critical problem in vaccine delivery systems, contrasting success with limitations.<br><br>Parallel structure pattern: "while the nanovesicles successfully deliver the genetic payload and the ability" implies completion of what nanovesicles lack (e.g., "to replicate" or "to be functional").<br><br>Final token "ability" is mid-phrase, requiring a gerund or to-infinitive completing the parallel contrast about what the nanovesicles cannot do alongside delivery success. |
| λ=0.002 | 75 | 0.108 | Scientific explanation of protein aggregation challenges in synthetic cell research.<br><br>Logical argument building toward consequences of protein unfolding: loss of function follows "ability to..."<br><br>Final phrase "ability to" is an incomplete infinitive construction requiring a verb describing protein functionality, e.g., "perform," "function," or "interact." |
| λ=0.006 | 29 | 0.085 | scientific explanation pattern<br><br>functionality of proteins<br><br>"ability to" requires completion |
| λ=0.015 | 22 | 0.509 | scientific explanation<br>protein function<br>"ability" |
| λ=0.03 | 15 | 0.211 | to retain their ability |

### Sample 27
> …ns other U.S.-detailed Chinese tech heavyweights these types of as Alibaba,

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 120 | 0.754 | List-completion pattern: the article enumerates tech giants supporting China's economy; Tencent and Alibaba have been mentioned, strongly implying another company (e.g., Baidu, JD.com) follows as the series continues.<br><br>Narrative momentum toward economic optimism: the text builds China's economic resilience through positive indicators, so additional supporting companies will reinforce this optimistic framing.<br><br>Final token "Alibaba," is a comma after the first listed tech giant, part of an ongoing enumeration; the next tokens must name another company in the same company-list structure. |
| control RL (λ=0) | 96 | 0.750 | Enumerated list of Chinese tech giants building data centers expected to continue (Alibaba just named, next entries like Tencent, Baidu, JD.com, or ByteDance are strongly implied).<br><br>Financial/business report genre establishing competitive landscape, requiring continued enumeration of Chinese companies with market relevance.<br><br>The final token is a comma after "Alibaba," mid-list, directly requiring additional named Chinese tech companies to follow in parallel structure. |
| λ=0.001 | 83 | 0.681 | Enumeration pattern: "Alibaba Cloud, Tencent and Alibaba" lists competing Chinese tech firms, expecting at least one more example.<br><br>Market analysis context: discussing tech firms expanding to Southeast Asia, driving competition narrative requiring additional companies or elaboration.<br><br>Final token is a comma after "Alibaba," mid-list, demanding continuation with another named competitor or company name. |
| λ=0.002 | 79 | 0.791 | Continuation of a list of Chinese tech giants with Alibaba as the first item, requiring completion with additional names like Tencent.<br><br>The sentence structure "Alibaba, [others]" signals an ongoing enumeration of companies.<br><br>The final token is a comma after "Alibaba," mid-list, requiring an additional company name or more names to follow. |
| λ=0.006 | 31 | 0.735 | Business news context<br>competitor list continuation<br>"Alibaba," requires another company name |
| λ=0.015 | 25 | 0.782 | tech stock list<br>major companies<br>"Alibaba," |
| λ=0.03 | 14 | 0.472 | Alibaba, |

### Sample 28
> … particulates. Close to 99 percent of particulates are removed by equipment

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 136 | 0.759 | Explanatory/informational pattern establishing a cause-and-effect explanation for the ash color problem, requiring completion of the removal process description.<br><br>Technical domain context: power plant emissions terminology ("air pollution controls," "scrubbers," "flue gas desulphurization") signals highly specialized environmental engineering vocabulary expected throughout.<br><br>Final token "equipment" ends an incomplete relative clause mid-thought — "by equipment" strongly anticipates a specific noun or noun phrase specifying what type of equipment (e.g., "that removes sulfur dioxide" or "called scrubbers"), continuing the causal explanation of how sulfur dioxide is reduced. |
| control RL (λ=0) | 142 | 0.822 | Technical/environmental report describing coal-fired power plant air quality processes, establishing factual prose style with precise terminology like "sulfur dioxide" and "fly ash."<br><br>Narrative momentum: the passage is explaining how emissions are controlled at the source, building toward completing the mechanism — "equipment" refers to specific air pollution control devices (scrubbers, electrostatic precipitators).<br><br>Final token "equipment" is a noun mid-sentence, part of a relative clause explaining how particulates and SO₂ are removed; the next tokens must name or describe the type of equipment (e.g., "at the plant" or "such as scrubbers"). |
| λ=0.001 | 85 | 0.813 | Industrial/technical description of coal plant emissions, establishing context for pollution control measures.<br><br>The phrase "removed by equipment" requires completion specifying what equipment or system (e.g., "located in the stack" or "at the plant").<br><br>Final token "equipment" ends an incomplete clause mid-sentence, demanding continuation describing where the equipment is located or what it does. |
| λ=0.002 | 69 | 0.809 | Environmental/energy article explaining emissions controls at coal power plants.<br><br>Explanation of coal ash removal process requiring completion of the technical mechanism.<br><br>Final token "equipment" ends an incomplete relative clause ("by equipment"), requiring a prepositional phrase like "at the plant" or "before discharge." |
| λ=0.006 | 30 | 0.383 | Environmental/energy article context<br>pollution removal explanation<br>"equipment" requires completion |
| λ=0.015 | 26 | 0.527 | coal-fired plant description<br>air pollutant removal<br>"equipment" |
| λ=0.03 | 12 | 0.274 | equipment |

### Sample 29
> …ou can add a little moisture if needed, but not too much. 3. Plant them up 

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 129 | 0.093 | Numbered list format ("1. Choosing the right seeds," "2. Choosing the right container") signals continuation with item "3. Sowing your seeds," requiring sequential garden tips with consistent heading + body structure.<br><br>Practical gardening guide tone — conversational yet instructional, with step-by-step advice on growing sunflowers, expects actionable next steps after the container section.<br><br>The word "planting" ends the sequence as the start of a new section, immediately requiring a period, elaborating sentence explaining how/when to plant seeds into the container, matching prior section's explanatory style. |
| control RL (λ=0) | 120 | -0.137 | Sequential gardening tutorial structure: each numbered/sectioned step explains a distinct phase in sowing peas, requiring continuation of practical instructions.<br><br>The text has been methodically covering preparation steps (soil preparation, sowing seed), implying the next logical step is germination, growth, or transplanting peas into containers.<br><br>Final token "Plant" is a new heading/section label (likely bold or sectioned), beginning a new procedural step in the sowing process — the next tokens should describe planting timing, depth, or method for containers. |
| λ=0.001 | 86 | -0.075 | Bulleted list structure continues, following the pattern of numbered steps; "Planting" signals the next step's description.<br><br>Cultivation guidance context requires practical, step-by-step gardening instructions for growing tomatoes from seeds.<br><br>Final token "Planting" is a section header/heading in the bullet list, immediately requiring a descriptive sentence explaining how to plant tomato seeds. |
| λ=0.002 | 70 | -0.074 | Sequential gardening tutorial structure: numbered steps guide planting process step-by-step.<br><br>The process of planting is mid-explanation, with "Once your seeds are sprouted" signaling the next actionable step.<br><br>Final token "Once" opens a conditional clause requiring a verb phrase describing the next planting step. |
| λ=0.006 | 38 | -0.015 | Garden plant growing guide<br><br>Next step in transplant process<br><br>"Once you have your seedlings" begins procedural next step |
| λ=0.015 | 28 | -0.049 | planting instructions<br>step-by-step process<br>"Once you're ready" |
| λ=0.03 | 11 | 0.134 |  |

### Sample 30
> …search and development. We are currently accepting applications for the 201

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 133 | 0.522 | News article structure signals a dateline/date stamp pattern: "October 22, 201" is a truncated year requiring 3 digits, almost certainly "201" → "2017" or similar modern year.<br><br>Formal science journalism register maintains objective, factual tone throughout, with institutional credit and technical terminology.<br><br>The final token "201" is an incomplete 4-digit year within a dateline, immediately constraining the next tokens to complete the year (e.g., "7" or "8"), likely followed by a comma and "Washington DC." |
| control RL (λ=0) | 154 | 0.547 | Acronym expansion pattern: "XeTH" was spelled out as "X-ray-enabled Tilted High Harmonic" in parentheses, so "XeTH-" is the acronym being repeated to continue the sentence.<br><br>News article format with citation/award context: the text is describing a recognized scientific achievement, establishing a pattern of naming the prize, the awarding body, and the year — "201" strongly signals a specific year completing the award context.<br><br>Final token "201" is the beginning of a year (likely "2015," "2017," or "2019"), mid-number, requiring immediate digits to complete the four-digit year. |
| λ=0.001 | 98 | 0.425 | Technical article about quantum metrology innovation, establishing credibility through citations and dates.<br><br>The article is concluding with a formal attribution citing the year of the breakthrough, requiring a specific four-digit year to complete.<br><br>"201" is a partial year mid-attribution, immediately constraining next tokens to complete "201X," likely "2013" given the "2 years ago" timestamp earlier in the text. |
| λ=0.002 | 77 | 0.507 | Institutional news article transitioning to publication credit/citation format.<br><br>Pattern of listing technical details, patents, and research details before concluding with acknowledgment.<br><br>"The 201" is a truncated year (201X), almost certainly completing as "201X," requiring a 4-digit year to follow. |
| λ=0.006 | 34 | 0.488 | scientific news article<br><br>publication citation required<br><br>"201" is year "2017" |
| λ=0.015 | 24 | 0.145 | scientific announcement<br>date reference<br>"111" |
| λ=0.03 | 14 | 0.140 | techno1 |

### Sample 31
> … Kristin Asbjørnsen Quartet will perform six concerts in Norway and Germany

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 133 | 0.622 | Biographical/professional CV/artist statement format establishes domain: career narrative with dates, institutions, and geographical movement.<br><br>Narrative momentum: the text has been detailing the artist's educational journey (USA → Germany → Austria) and is transitioning to career achievements, with "performances at... Germany" beginning a list of international events or exhibitions.<br><br>Final token "Germany" closes a location phrase ("in Berlin and Germany") within an ongoing sentence listing professional engagements — next tokens will complete or extend this list, likely continuing with a comma, further locations, or a verb describing what happened during these performances. |
| control RL (λ=0) | 118 | 0.751 | Press release format signals an upcoming list of dates/places, typical of tour announcements following "Germany" as a destination country.<br><br>Narrative momentum: the release has described the CD's contents, reception, and now concludes with tour logistics, requiring continuation of event details.<br><br>Final token "Germany" is part of a list of locations ("in Austria and Germany"), requiring a preposition or time phrase (e.g., "in 2023," "and will take place in...") to complete the tour announcement sentence. |
| λ=0.001 | 97 | 0.791 | Biographical profile transitioning to performance tour details, establishing geographic scope of the project.<br><br>The phrase "Germany" ends a list of tour countries ("in Italy and Germany"), suggesting continuation with additional details like dates, venues, or further collaborators/performances.<br><br>Final token "Germany" is mid-sentence after "and," requiring immediate completion of the country pair, likely followed by a prepositional phrase or additional tour logistics. |
| λ=0.002 | 72 | 0.603 | Biographical/professional artist profile establishing career scope and geography.<br><br>List continuation pattern: "Germany" implies further country details or tour dates expected next.<br><br>Final token "Germany" is the end of a geographic scope list ("Germany"), likely followed by additional locations, dates, or a tour description. |
| λ=0.006 | 32 | 0.718 | biographical artist statement<br>tour/venue details expected<br>"and Germany" completes location list |
| λ=0.015 | 22 | 0.553 | tour announcement<br>dates coming<br>"Germany" |
| λ=0.03 | 13 | 0.315 | in Germany |

### Sample 32
> … want to edit directly the /boot/grub/menu.lst you can download the package

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 129 | 0.716 | Technical documentation pattern: step-by-step guide for configuring IPv6 static routes on Debian/Ubuntu, maintaining instructional tone throughout.<br><br>Ongoing instruction momentum: the guide is mid-explanation of a method for adding multiple routes; the sentence beginning "The way I did that was to install the package" promises a specific package name or command to follow.<br><br>Final token "package" is the subject of an incomplete clause ("to install the package"), immediately requiring a package name or description — likely a comma then a specific Debian/Ubuntu package name like "net-tools" or "iproute2". |
| control RL (λ=0) | 117 | 0.694 | Instructional/tutorial genre with procedural steps being listed, promising a solution for enabling "aptitude" functionality in Debian/Ubuntu.<br><br>The text builds toward a specific software tool/package name, as "the package" promises a named product (likely a Debian/Ubuntu GUI or GUI installer package like `debconf` or `debconf-utils`).<br><br>Final token "the package" is a noun phrase opener requiring an immediate noun — the specific Debian package name being recommended, likely a GUI-related package that enables GUI-based package configuration. |
| λ=0.001 | 80 | 0.679 | Technical instructions on configuring Ubuntu's clock settings, establishing context for the next step.<br><br>The phrase "install the package" signals an immediate software recommendation follows, specifying which package (likely `tzdata`).<br><br>Final token "package" ends mid-sentence after "install the package," requiring the package name and possibly installation command to follow immediately. |
| λ=0.002 | 80 | 0.704 | Technical how-to guide explaining Linux GRUB resolution for French locales<br><br>Step-by-step instructional momentum: the writer is mid-instruction giving solution details<br><br>"the package" ends mid-sentence, immediately requiring a specific package name (e.g., `grub`, `grub-legacy`, or `grub2`) to follow |
| λ=0.006 | 30 | 0.567 | technical installation guide context<br><br>software package reference<br>"the package" requires name continuation |
| λ=0.015 | 24 | 0.524 | software installation<br>Debian package<br>"the package" |
| λ=0.03 | 13 | 0.272 | the package |

### Sample 33
> …Every tree is remarkable because even one can curb greenhouse gases and cut

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 124 | 0.760 | Narrative momentum: article builds from local initiative to broader context, now transitioning to environmental/ecological benefits of urban trees.<br><br>Causal explanation pattern: "trees also help reduce air pollution and cut" sets up a paired list structure ("and cut...") requiring a parallel noun phrase like "air pollution" or "traffic congestion."<br><br>Final token "cut" is a transitive verb beginning a prepositional phrase (e.g., "cut down air pollution," "cut down on traffic"), requiring a direct object to complete the parallel with "reduce air pollution." |
| control RL (λ=0) | 126 | 0.791 | Ongoing description of urban trees benefits: "cut" strongly anticipates a verb phrase like "cut down" or "cut down air pollution/air temperature," continuing the list of environmental benefits trees provide.<br><br>The article follows a news-press release format establishing urban trees' environmental value, with escalating benefit claims ("cool," "clean air," "pollution").<br><br>Final token "cut" is a verb beginning a verb phrase mid-sentence, immediately requiring a direct object or complement — most likely "cut down" or "cut air pollution," completing the benefit clause. |
| λ=0.001 | 97 | 0.782 | Bullet-point list pattern with parallel structure: each item describes street trees' environmental benefits.<br><br>The text is enumerating ecological benefits of urban trees, so remaining benefits (air quality, carbon sequestration, heat reduction) are expected.<br><br>Final fragment "can cut" is mid-phrase, requiring completion of a verb phrase describing how trees reduce pollution or heat, e.g., "air pollution" or "heat island effects." |
| λ=0.002 | 81 | 0.780 | News article reporting on community tree-planting initiative, maintaining informational journalistic style.<br><br>List of benefits being enumerated: "save energy and cut" implies continuation of environmental advantages (e.g., "air pollution," "carbon emissions").<br><br>Final token "cut" ends an incomplete phrase listing tree benefits, requiring a noun phrase describing a reduction or decrease. |
| λ=0.006 | 31 | 0.559 | environmental benefits list<br>quantified pollution reduction<br>"and cut" continues parallel structure |
| λ=0.015 | 25 | 0.604 | environmental benefits<br>tree benefits list<br>"can cut" |
| λ=0.03 | 13 | 0.137 | can cut |

### Sample 34
> …tic neuroendocrine tumor (PanNET) to study SSTR2-based radionuclide imaging

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 159 | 0.483 | Scientific/technical paper abstract format: formal academic register established, expecting continuation of methodology and findings. The phrase "using 89Zr-labeled N-(2-hydroxypropyl)-β-cyclodextrin as an MRI-based imaging" signals a contrast or elaboration is coming.<br><br>Argumentative momentum: the abstract sets up a contrast between PET's specificity and MRI's poor resolution, building toward presenting the novel compound's imaging capabilities as a solution.<br><br>Final token "imaging" ends a noun phrase mid-construction, strongly constraining the next token to continue the noun phrase (e.g., "agent," "marker," "probe") or begin a relative clause/verb phrase describing the compound's role. |
| control RL (λ=0) | 143 | 0.587 | Scientific paper abstract structure: the text introduces a study methodology (Gd/HPA-PEG nanoparticles) and its purpose, establishing the "methods" section framing of an abstract.<br><br>Domain-specific biomedical research context: nanoparticle labeling, MRI contrast agents, and tumor detection vocabulary constrain next tokens to technical terms like "and therapy," "of tumors," or "in vivo."<br><br>Final token "imaging" closes the phrase "noninvasive radioisotope labeling for MRI contrast agent imaging," which is mid-abstract — the sentence likely continues with "and therapy" or "of tumor detection" or a period/next sentence beginning with findings/results. |
| λ=0.001 | 96 | 0.626 | Scientific abstract structure: methods/results expected after stating purpose of radiotracer development.<br><br>Domain-specific terminology momentum: nuclear medicine imaging context requires continuation with methodology details like radiolabeling, tumor models, or imaging techniques.<br><br>Final token "imaging" ends an infinitive phrase "for radiotracer-based imaging," immediately constraining next tokens toward specification of experimental models, results, or further elaboration on applications. |
| λ=0.002 | 69 | 0.523 | Scientific paper abstract structure: introduction of problem and methods signals results/findings next.<br><br>Narrative momentum toward experimental outcomes — the study's purpose drives expectations of results.<br><br>Final token "imaging" ends a clause introducing a goal phrase; immediate continuation likely specifies outcomes or methodology details. |
| λ=0.006 | 35 | 0.573 | Research methods description<br><br>imaging application context<br><br>"for radionuclide imaging" ends mid-phrase |
| λ=0.015 | 26 | 0.491 | scientific study<br>radiotherapy<br>"radioligand imaging" |
| λ=0.03 | 17 | 0.241 | for radionuclide imaging |

### Sample 35
> …asic aluminium chloride, polybasic aluminium chloride, aluminium hydroxide,

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 141 | 0.683 | Industrial chemical catalog/market report format, listing product descriptions, company names, and specifications for polymeric materials like PAM, PAC, and polyacrylates.<br><br>Repetition/continuation pattern: the text cycles through company name introductions, product descriptors, and technical specifications for water treatment chemicals; another product name or compound is expected to complete the list of polyacrylic acid derivatives.<br><br>Final token "polyacrylic acid hydrate," is mid-list after "polyacrylic acid sodium, polyacrylic acid magnesium, polyacrylic acid aluminium hydroxide," — a comma signals an additional compound name or descriptor must immediately follow. |
| control RL (λ=0) | 123 | 0.665 | Product description/chemical listing format: the text has shifted from article body to a bulleted product description listing chemical names, requiring continuation of this pattern.<br><br>List continuation pattern: several compound names have been listed with commas ("aluminum hydroxide, aluminum oxide, aluminum sulfate, aluminum hydroxide"), implying more entries follow with similar chemical names.<br><br>Final token is a comma after "aluminum hydroxide," signaling an incomplete list; the next token must be another chemical compound name (likely aluminum-related, given the "aluminum" repeating pattern). |
| λ=0.001 | 93 | 0.692 | Chemical/nomenclature enumeration: listing synonyms for aluminum hydroxide continues with more compound names.<br><br>Structured database/metadata pattern: product listing with formula, synonyms, supplier info, and synonyms list suggests more synonyms follow.<br><br>Final token "aluminum hydroxide," is mid-list within synonyms, requiring additional compound names like "alumina" or "aluminum oxide" to complete the enumeration. |
| λ=0.002 | 65 | 0.691 | Continuation of a list of aluminium compound chemical names separated by commas.<br><br>Database/FAQ content pattern with structured product listing and keyword enumeration.<br><br>Final token "hydroxide," is mid-list with a comma, strongly constraining next token to another aluminium compound name. |
| λ=0.006 | 33 | 0.579 | Chemical list continuation<br><br>product catalog format<br><br>"aluminum hydroxide," needs more chemicals |
| λ=0.015 | 28 | 0.663 | chemical names list<br>inorganic compound<br>"aluminum hydroxide," |
| λ=0.03 | 16 | 0.550 | aluminum hydroxide, |

### Sample 36
> …d by the European Agricultural Fund for Rural Development. Project duration

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 125 | 0.854 | Project description follow-up: the text transitions from project scope to specific details, signaling enumeration of technical specifications or timeline information.<br><br>Structured project documentation pattern: formal tender/contract language with section headers like "Project duration" implies a list of standard project sections (e.g., scope, budget, deliverables) will follow.<br><br>"Project duration" is a section header, and immediately constrains next tokens to be a time frame, deadline, or start/end dates — e.g., "was 12 months" or "started in July 2023." |
| control RL (λ=0) | 122 | 0.858 | Project documentation structure: sections follow title/section header patterns, implying a table of contents or formal project summary format.<br><br>"Project duration" is a section header or field label, requiring an immediately following colon, number, or time expression like "Project duration: 12 months" or "Project duration: November 20XX to April 20XX."<br><br>The final token "duration" is part of an unclosed section heading "Project duration," acting as a label that demands a colon followed by a time period or time interval value. |
| λ=0.001 | 97 | 0.871 | Project specification document listing technical details, establishing pattern of labeled field entries like "Water pressure," "Water flow."<br><br>Narrative momentum: after detailing system components and pressures, duration/contractual timeline is next logical field.<br><br>"Project duration" is the final entry, a section label requiring an immediate value or timeframe to follow, e.g., "36 months" or "2021-2023." |
| λ=0.002 | 70 | 0.876 | Structured project report format with labeled sections driving content expectations.<br><br>List of project fields being enumerated sequentially (Location, Project scope, Project duration...).<br><br>"Project duration" is a section header, immediately requiring a specific time period/length response like "24 months" or "X months." |
| λ=0.006 | 29 | 0.836 | Project description format<br><br>Project duration section header<br><br>"Project duration" needs completion |
| λ=0.015 | 22 | 0.728 | project description<br>timeline<br>"Project duration" |
| λ=0.03 | 13 | 0.517 | Project duration |

### Sample 37
> …motive's and be sent back home. (Not real) R models were only produced in 1

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 125 | -0.031 | Automotive enthusiast forum content mixing personal anecdotes with product reviews, establishing a pattern of casual, conversational tone with enthusiastic exclamations.<br><br>The text is mid-sentence introducing a specific car model example ("2020 Ford Mustang GT"), suggesting continued automotive content referencing Mustang-specific features, performance specs, or a follow-up product description.<br><br>Final token "gt" is a variant of "GT" (Gran Tourismo), a proper noun completing a model designation, immediately constraining what follows to either a new sentence continuing the Mustang GT topic or further specification details. |
| control RL (λ=0) | 135 | 0.533 | Repetition/continuation pattern: "Mustang" appears again, mirroring the earlier text's repeated noun usage — the series references "this particular Ford Mustang" and "the new Mustang."<br><br>Narrative momentum: the sentence is mid-completion, describing the car being reviewed ("this particular Ford Mustang"), so the next tokens will continue with details about its appearance, trim, or features.<br><br>Final token: "1" follows "Mustang," beginning a model-year designation (e.g., "1970"), a year or specific model descriptor — the immediate next token is almost certainly a year number. |
| λ=0.001 | 130 | 0.378 | The text is mid-sentence describing a 2022 Nissan GT-R NISMO's production specs, requiring completion of the technical claim.<br><br>The article follows a pattern of listing car features/specs with precise figures (e.g., "5.2 liters," "800 hp"), expecting similar specific data.<br><br>Final token "the 2" begins a model designation or technical specification phrase, likely completing "the 2022 Nissan GT-R NISMO" or a horsepower figure like "the 2022 Nissan GT-R NISMO produces." |
| λ=0.002 | 90 | 0.249 | List-style article continuing with Ford Mustang facts, establishing a factual enumeration pattern.<br><br>Incomplete sentence mid-thought: "They also own a particular 1" signals a model specification is being quoted.<br><br>The final token "1" is the beginning of a Ford Mustang trim/model number (e.g., "125 S" or "1965"), requiring numeric or alphanumeric continuation. |
| λ=0.006 | 34 | 0.205 | Car show review narrative<br>specific model details expected<br>"this particular car is a" requires model name |
| λ=0.015 | 25 | 0.264 | car review<br>specific model name<br>"this 1" |
| λ=0.03 | 14 | 0.008 | that 1 |

### Sample 38
> …sar, Shanthanu Bhagyaraj, Sriman, Andrea Jeremiah, Sanjeev, Srinath, VJ Ram

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 142 | 0.608 | Celebrity gossip blog post format, with structured sections like "Read more" links and taglines establishing a repetitive, formulaic entertainment news article style.<br><br>A film production update is being detailed — "Kavithaiyum Karupattiyum" (K.K.K) has its cast listed, with two names given so far, strongly implying more names in the same list structure.<br><br>The final token "Ram" is mid-cast-list within parentheses "(Nithiyakumar and V J Ram-", a truncated actor name requiring completion (likely "Ramakrishnan" or similar surname) followed by closing punctuation to close the parenthetical. |
| control RL (λ=0) | 123 | 0.673 | Celebratory social media post pattern: upbeat tone, multiple cast mentions, song credits, and now a director credit following industry convention.<br><br>The director's name "Vijay Ram" is incomplete — a proper name mid-word or mid-credit requiring immediate completion, likely with surname followed by role/song reference.<br><br>Final token "Ram" is the partial name of a Tamil film industry director "Vijay Ram" (possibly Vijay Rao or a similar name), requiring the surname or a continuation specifying the song/album this credit pertains to. |
| λ=0.001 | 91 | 0.636 | Celebrity gossip/reality TV article structure signals an announcement of an upcoming collaboration with Vijay Ramesh.<br><br>The incomplete name "Vijay Ram" requires its suffix "esh" to complete, likely followed by details of the collaboration.<br><br>Final token "Ram" is mid-name in a collaboration announcement, immediately constraining next tokens to complete "esh" and then describe the joint project. |
| λ=0.002 | 68 | 0.601 | List of Tamil film names being enumerated sequentially after "Vijay, R Vijay and Vignesh"<br><br>Continuation of film director's name completion<br><br>"Vijay Ram" is a partial Tamil film title/producer name requiring completion of the full title name next. |
| λ=0.006 | 36 | -0.207 | Tamil cinema review context<br>names completion expected<br>"Vijay and Vijay Ram" requires full name |
| λ=0.015 | 30 | 0.739 | cast listing<br>"Vijay and VJ Ram"<br>"VJ Ram" |
| λ=0.03 | 14 | 0.011 | Vikram Ram |

### Sample 39
> …, ACCA as well as relevant experience in international banking. In addition

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 93 | 0.718 | Job posting genre following standard HR format: qualifications, skills, then additional requirements — next section continues listing requirements/conditions.<br><br>Continuation of structured qualifications/requirements pattern: each bullet or clause introduces a new criterion (age, marital status, nationality, educational background, etc.).<br><br>Final token "In addition" opens a conjunction clause mid-sentence, requiring continuation listing another qualifying condition or criterion for the position. |
| control RL (λ=0) | 100 | 0.726 | Job posting listing qualifications, expecting continuation of requirement list with structured bullet points or phrases.<br><br>The "In addition" signals a list continuation pattern, implying more job requirements (e.g., skills, experience, education) are about to be listed alongside or after existing qualifications.<br><br>"In addition" is a transitional phrase mid-sentence, requiring an immediate continuation—likely a comma or clause introducing another qualifying criterion or a new bullet point describing further requirements. |
| λ=0.001 | 69 | 0.724 | Job posting format signals a list of qualifications/requirements continuing after "In addition."<br><br>Parallel structure expected: "In addition" mirrors the opening requirement list, implying further criteria follow.<br><br>Final token "In addition" is a transition phrase opening an additive clause, requiring continuation of job qualifications. |
| λ=0.002 | 47 | 0.606 | Job listing format with qualifications section continuing after "In addition"<br><br>List of required skills and attributes is expected to follow<br><br>"Additionally" signals continuation of comma-separated qualification criteria |
| λ=0.006 | 28 | 0.614 | Job description format<br><br>qualification criteria continuation<br><br>"Additionally" signals more requirements |
| λ=0.015 | 22 | 0.629 | job posting<br>qualifications<br>"In addition" |
| λ=0.03 | 12 | 0.346 | In addition |

### Sample 40
> …iewed from this source for any purpose, but reproduction or distribution in

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 122 | 0.081 | Academic journal metadata format, with structured fields (e.g., "Date: June 13, 2019", "Number: 1") establishing a bibliographic record template.<br><br>The copyright/reproduction restrictions notice follows a standard institutional pattern, typically ending with prohibited reproduction details like "any form" or "any medium" or specific restrictions.<br><br>Final token "in" begins a prepositional phrase listing permitted/restricted reproduction contexts; immediate next tokens expected are "any form," "any medium," or similar restricted reproduction language. |
| control RL (λ=0) | 125 | 0.062 | Academic metadata record format (bibliographic citation followed by repository permissions notice) dictates standard institutional repository language.<br><br>The phrase "distribution in" is a partial sentence in a copyright/repository permissions statement, strongly constraining next tokens to a prepositional phrase like "any form" or "any media," followed by permitted use conditions.<br><br>Final token "in" is a preposition opening a prepositional phrase within the repository permissions clause, immediately requiring a noun phrase such as "any form" or "any media" to complete the standard academic repository disclaimer. |
| λ=0.001 | 88 | 0.369 | Legal/academic citation boilerplate pattern establishing copyright permissions and restrictions on reproduction.<br><br>The phrase "distribution in" signals an incomplete list of prohibited reproduction methods (e.g., "any form or by any means").<br><br>Final token "in" is mid-phrase, requiring completion of the distribution restriction clause, likely continuing with "any form or by any means" or similar legal formula. |
| λ=0.002 | 74 | 0.227 | Academic repository metadata format with copyright/distribution policy section established.<br><br>Legal distribution restriction language is standard in institutional repository terms.<br><br>"distribution in" is the beginning of a legal phrase, most likely completing as "distribution in any medium or format, other than for..." or "distribution in any form other than..." |
| λ=0.006 | 26 | 0.016 | Academic repository metadata<br><br>Reproduction policy statement<br><br>distribution in |
| λ=0.015 | 23 | 0.071 | academic repository<br>distribution rights<br>"distribution in" |
| λ=0.03 | 13 | -1.140 | distribution in |

### Sample 41
> …ed lens - True day/night - OSD menu - Smart IR, EXIR 2.0 technology - Up to

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 137 | 0.717 | Product specification list pattern: structured list of bullet points detailing technical parameters (e.g., "- 1920 × 1080p", "- 1.5 million pixels") establishing continuation format.<br><br>Domain-specific technical specs: each bullet follows a consistent pattern of "- [feature]: [value]" describing camera hardware capabilities, suggesting more specs like resolution, connectivity, or interface follow.<br><br>"Up to" is a comparative phrase opening a specification range or maximum value, immediately requiring a numerical figure or dimension (e.g., "160 meters," "2.4 million pixels") to complete the spec. |
| control RL (λ=0) | 119 | 0.635 | Product specification document listing technical features of a smart camera, establishing a pattern of feature → value pairs with colon-separated values.<br><br>The document has enumerated several capabilities (night vision, intelligent recognition, motion tracking, remote monitoring, 3D imaging), suggesting more specs or a concluding feature description follows.<br><br>"Up to" is an incomplete prepositional phrase introducing a numeric or quality limit, immediately requiring a specific value (e.g., "50 meters" or "1,000 frames per second") to complete the specification. |
| λ=0.001 | 86 | 0.668 | Product specification list format, detailing technical features of surveillance equipment with bullet-style attributes.<br><br>Continuation of performance specifications — "Up to" begins a maximum value statement, requiring a numeric figure followed by a measurement unit (e.g., "1200m").<br><br>Final token "to" completes "Up to," immediately requiring a specific range or performance figure to follow. |
| λ=0.002 | 77 | 0.672 | Product specification list pattern: technical bullet points detailing camera features.<br><br>Numeric capability completion expected: "Up to" signals a maximum value like resolution or distance.<br><br>"Up to" is an incomplete phrase requiring an immediate quantified specification (e.g., "200m," "4K") to complete the feature description. |
| λ=0.006 | 30 | 0.440 | Technical specification list<br>maximum distance specification expected<br>"Up to" requires distance value |
| λ=0.015 | 23 | 0.510 | technical specs<br>camera features<br>"Up to" |
| λ=0.03 | 13 | -0.174 | Up to |

### Sample 42
> …often precedes visible pathological manifestation. PhD Molecular Plant Path

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 157 | 0.385 | Structured table of contents with hierarchical section headers (e.g., "1. The Cell, 2.1. Structure and Organisation, 2.2. Plant Virology") — a consistent two-digit/subordinate-numbering pattern dominates, predicting further subsection entries.<br><br>The document is a textbook index/reference list for Plant and Environmental Science students, maintaining formal academic vocabulary throughout with precise subject-specific terminology.<br><br>Final token "Path" is a truncated word mid-table entry, part of the sequence "3. Microbial Path" — almost certainly completing as "ology" or "ology and Plant Immune Responses," consistent with the preceding section label "3. Microbial Pathology and Plant Immune Responses" already seen above. |
| control RL (λ=0) | 113 | 0.524 | Academic conference program listing format establishes a structured agenda pattern, with event details, speaker names, and session titles following a consistent format.<br><br>The list of speaker sessions is mid-enumeration; multiple named sessions with speaker names in parentheses are being listed, following the pattern "(Name) [Session Title]."<br><br>"Microbiology of Plant Path" is a truncated session title (likely "Plant Pathology"), mid-word, requiring immediate completion of that compound noun phrase before the session description or further details follow. |
| λ=0.001 | 103 | 0.612 | Institutional/academic profile format continues, listing research groups and expertise of the John Innes Centre.<br><br>The sentence about "Research into Molecular Plant Path" is mid-phrase, requiring completion of the field name (likely "ology" or "ogenesis").<br><br>Final token "Path" is a truncated word in a research theme description, immediately requiring completion of the compound noun, e.g., "ology" or "ogenesis" followed by further research detail. |
| λ=0.002 | 71 | 0.615 | Academic conference announcement following a structured format with speaker bios and session details.<br><br>List of departmental or subject names continuing in sequence after "Microbial Plant Path"<br><br>Final token "Path" is the truncated end of "Microbial Plant Pathology," requiring immediate completion of this word/phrase. |
| λ=0.006 | 32 | 0.550 | Academic profile structure<br><br>Research interests listed<br><br>"Plant and Microbial Path" is truncated |
| λ=0.015 | 26 | 0.522 | academic profile<br>research interests<br>"Plant and Microbial Path" |
| λ=0.03 | 13 | 0.158 | Plant Path |

### Sample 43
> … design. Measurements of piston temperature during ICEs operation represent

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 147 | 0.746 | Academic dissertation abstract structure: after introducing methodology and data, the text transitions to a discussion/contribution statement, signaling elaboration on thermal measurements' importance.<br><br>Narrative/argumentative momentum: the abstract builds from motivation (reliability importance) to methodology to now addressing a specific research gap — "measurements on the wall temperature represent" sets up a definitional or evaluative claim about their significance or challenge.<br><br>Final token "represent" is a linking verb beginning a predicate, requiring a noun phrase or adjective complement immediately next — likely describing what wall temperature measurements represent in the context, e.g., "a critical diagnostic tool" or "an essential but challenging parameter." |
| control RL (λ=0) | 124 | 0.752 | Academic thesis abstract establishing methodology context for measurement techniques; formal academic register maintained throughout.<br><br>Logical argumentative structure: the passage has introduced the research problem (temperature gradient modeling) and now transitions to justifying the study — measurement methods are being introduced as the means to solve this problem.<br><br>"represent" is a verb beginning a predicate that requires a noun phrase complement; given the scientific/academic context, the next tokens must describe what temperature measurements "represent" in this domain — likely something like "a critical/challenging aspect" or "an essential component." |
| λ=0.001 | 87 | 0.608 | Academic paper introduction establishing motivation and research gap, requiring elaboration of measurement challenges.<br><br>The phrase "represent" signals an imminent predicate completing a noun clause about what surface temperature measurements mean/signify in combustion research contexts.<br><br>Final token "represent" is a linking verb mid-sentence, immediately requiring a noun phrase or adjective complement describing the significance/importance of surface temperature measurements. |
| λ=0.002 | 66 | 0.737 | Academic thesis introduction establishing research scope and problem context.<br><br>The sentence sets up a contrast between known boundary conditions and unknown surface temperatures.<br><br>"The measurement of wall surface temperatures represent" is an incomplete predicate requiring a noun phrase or adjective completing the description of their significance/challenge. |
| λ=0.006 | 31 | 0.420 | Academic paper structure<br><br>technical domain: combustion diagnostics<br><br>"represent" requires completing description |
| λ=0.015 | 28 | 0.549 | technical paper<br>thermocouples<br>"thermocouples represent" |
| λ=0.03 | 12 | 0.069 | represent |

### Sample 44
> … Groups Want to join the conversation? Please or to comment on this article

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 111 | 0.366 | Journalistic/informational article about women and mental health in construction, establishing a research-supported problem-solution structure throughout.<br><br>The article builds to a call-to-action section (comments box with prompt), signaling a closing engagement prompt to readers.<br><br>Final token "article" completes a standard comments section prompt phrase ("Add your own comment to this article"), which is the conventional closing formula for opinion/comment sections; next tokens likely begin a new paragraph, additional comment form fields, or a related article/link. |
| control RL (λ=0) | 120 | 0.403 | Journalistic/news article structure with a "comment box" feature — the article is closing with a call-to-action prompt for reader engagement.<br><br>The phrase "Add your thoughts to this article" follows standard newspaper commenting platform conventions, typically followed by a link or button label like "Leave a comment" or "Submit comment."<br><br>"this article" is the final phrase, completing a participial/call-to-action clause; the immediate next token is almost certainly a verb or link such as "Leave a comment," "Comment," or "below." |
| λ=0.001 | 86 | 0.421 | Journalistic/news article structure ending with a reader engagement prompt, signaling interactive commenting functionality.<br><br>The text follows a public health policy article with embedded call-to-action commentary, typical of policy journalism platforms.<br><br>Final token "article" closes the comment invitation phrase — next tokens likely include punctuation/period and further UI/CTA elements like "or email us at [contact]." |
| λ=0.002 | 61 | 0.444 | News article structure nearing end: comment invitation typical of journalism platforms like The Guardian.<br><br>Formal news register with embedded reader interaction call-to-action.<br><br>Final token "article" closes a comment-disclaimer sentence, strongly suggesting a link or punctuation follows. |
| λ=0.006 | 29 | 0.423 | Call-to-action completion<br>Comment submission prompt<br>Final preposition "this article" |
| λ=0.015 | 29 | 0.322 | comment section<br>read more link<br>"Read more to comment on this article" |
| λ=0.03 | 16 | -0.207 | to comment on this article |

### Sample 45
> …t 2 weeks later i guess i got that phase out of me, cause i switched to the

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 106 | 0.645 | User is sharing a personal device setup preference, contrasting two UI options, implying a positive recommendation is underway.<br><br>The structure "instead of [X] I use [Y]" signals a contrast favoring Android's default clock interface over the mentioned alternative (Faceroid).<br><br>Final token "the" opens a noun phrase describing what Android provides — the next word must be a specific UI component or app name, likely "default clock" or similar interface reference completing the contrast. |
| control RL (λ=0) | 114 | 0.637 | User is describing a mobile app upgrade/downgrade process, establishing a before-and-after narrative about UI/UX changes on a touchscreen interface.<br><br>The phrase "I was on the" sets up a contrast requiring a specific UI element name — the default lock screen interface the user previously used before switching.<br><br>Final token "the" is a definite article beginning a noun phrase, immediately requiring a specific named UI element (e.g., "lock screen," "stock lock screen," or similar) that the user now prefers. |
| λ=0.001 | 100 | 0.664 | Personal tech blog style established, sharing a phone upgrade anecdote with casual, conversational tone.<br><br>Narrative momentum: the author has upgraded to Droid Eris, pivoted to talk about the lock screen, implying a contrast or transition from previous lock screen experience.<br><br>Final token "the" begins a noun phrase specifying what replacement feature was used — next tokens must name the specific lock screen alternative (likely "old" or "default"). |
| λ=0.002 | 69 | 0.728 | User describing Android widget preferences, moving from one option to another.<br><br>Narrative pattern of listing tried widgets/options in sequence, implying a second choice being described.<br><br>"switched to the" is a transition phrase mid-sentence requiring a noun phrase naming the specific widget interface being replaced. |
| λ=0.006 | 30 | 0.446 | Smartphone customization context<br>replacement feature explanation<br>"the" requires a noun phrase |
| λ=0.015 | 25 | 0.611 | phone customization<br>status bar widgets<br>"changed to the" |
| λ=0.03 | 14 | 0.322 | now using the |

### Sample 46
> …a short drive to Joshua Tree National Park, the Town of Yucca Valley, and P

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 112 | 0.768 | Ongoing narrative review of a Las Vegas travel experience, listing sequential activities at the resort's golf course (putting green, practice range, driving range).<br><br>Repetition/continuation pattern: each activity described with action → outcome structure ("We played... We hit...").<br><br>Final token "P" is the beginning of a place name (likely "Parker" or similar location), completing the phrase "drive from downtown Las Vegas to P[arker]," part of a geographic description mid-sentence. |
| control RL (λ=0) | 127 | 0.816 | Geographic/sport tourism description pattern: each location highlights amenities, attractions, and nearby points of interest, building a comprehensive profile.<br><br>Enumeration of local attractions around Pinal Lake is mid-list, following the established pattern of naming nearby locations with distance and amenities ("15 miles from Pinal Lake, 30 miles from Sunizona and 50 miles from Phoenix").<br><br>Final token "P" is the start of a truncated place name, almost certainly "Phoenix" — continuing the geographic proximity enumeration ("50 miles from Phoenix"), requiring completion of this city name next. |
| λ=0.001 | 88 | 0.880 | Travel blog narrative building toward describing activities at Parker Valley campsite near Palm Springs.<br><br>List/continuation pattern: "and P" begins a continuation of nearby locations or activities, likely "Palm Springs" or "Pioneertown."<br><br>Final token "P" is the start of a proper noun mid-phrase ("and P"), requiring immediate completion of that location name. |
| λ=0.002 | 81 | 0.807 | Trip review genre with personal travel narrative, listing destinations and experiences sequentially.<br><br>The review is enumerating stops and attractions on a multi-city trip, building toward a concluding activity.<br><br>Final token "P" is the beginning of a place name (likely "Petaluma" or similar California location), mid-word, requiring completion of the place name. |
| λ=0.006 | 37 | 0.744 | trip review narrative momentum<br>location description completing<br><br>"and P" is "and P" beginning "Pine Mountain" |
| λ=0.015 | 21 | 0.456 | trip review<br>location names<br>"P" |
| λ=0.03 | 11 | 0.335 | P |

### Sample 47
> …loying capital in the sector. We hope Canadian companies will be leaders in

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 105 | 0.834 | Narrative momentum: article builds toward a forward-looking conclusion about Canadian companies leading in space mining/satellite technology.<br><br>Domain context: technical satellite/communications industry writing with policy implications, establishing Canadian leadership in emerging space domains.<br><br>Final token "in" opens a prepositional phrase completing "Canadian companies will be leading in [Noun Phrase]," requiring a specific domain noun like "space mining," "satellite communications," or "developing next-generation satellite technology." |
| control RL (λ=0) | 122 | 0.797 | Narrative momentum toward a forward-looking conclusion: the text has built a case for Canada's space industry strengths, now pivoting to a future prediction.<br><br>Domain-specific industry focus: the passage consistently centers on satellite technologies, space exploration, and Canadian innovation in space commerce, constraining the next tokens to those topics.<br><br>Final phrase "Canadian leaders will lead the way in" is an incomplete prepositional phrase requiring a specific domain noun phrase — likely "space," "this field," or "space exploration/commerce," completing the triumphant closing claim. |
| λ=0.001 | 93 | 0.863 | Narrative momentum: argument building toward Canada's leadership in space mining/commercial space, requiring a concluding assertion.<br><br>Domain context: space law and business strategy discussion drives expectation of forward-looking claims about Canada's role.<br><br>Final phrase "will likely be leaders in" is an incomplete predicate requiring a noun phrase specifying what domain or activity Canada leads — likely "space mining," "space resource utilization," or similar. |
| λ=0.002 | 75 | 0.792 | Argumentative momentum: article builds toward a concluding statement about Canadian leadership in space mining.<br><br>Thematic consistency: throughout emphasizes Canada's pioneering role and competitive advantage in space resource development.<br><br>Final token "in" opens a prepositional phrase requiring a noun phrase, likely "space mining" or "the space economy." |
| λ=0.006 | 35 | 0.728 | expert quote continuation<br>leadership in space mining<br>"early leaders in" requires a noun phrase describing domain |
| λ=0.015 | 27 | 0.746 | Canadian space industry<br>Canadian leadership<br>"likely to be leaders in" |
| λ=0.03 | 16 | 0.665 | likely to be pioneers in |

### Sample 48
> …sorted colors designs. If you want to know more about the specific product,

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 134 | 0.663 | Product listing format with structured specifications (name, price, description, details) signals a closing call-to-action section is beginning. The text follows a commercial retail page pattern.<br><br>The phrase "If you have any other questions about the product," is a standard CTA clause that almost always completes with a suggestion like "feel free to ask" or "please contact us."<br><br>Final token "product," is the end of a conditional subordinate clause opening; grammatically and commercially, the immediate next tokens must provide the main clause's response, e.g., "please contact us" or "feel free to reach out." |
| control RL (λ=0) | 118 | 0.680 | E-commerce product listing format with specifications, materials, and usage instructions — a standard "if you have more questions" closing prompt is expected next.<br><br>The phrase "If you have any questions regarding the product" is a classic e-commerce boilerplate call-to-action, strongly implying a follow-up clause about contacting support or visiting the store.<br><br>Final token is a comma after "product," continuing a conditional clause mid-sentence; the next token must complete this clause, likely "please contact us" or "write us an email/query." |
| λ=0.001 | 100 | 0.657 | Promotional/informational product page structure signals a closing call-to-action or contact inquiry section is underway.<br><br>The phrase "If you have any other questions regarding this product" sets up a conditional clause requiring completion with a recommended action (e.g., "please contact us" or "visit our website").<br><br>Final token "product," ends a subordinate clause mid-sentence, demanding an immediate main clause continuation like "please contact our support team." |
| λ=0.002 | 69 | 0.386 | Customer service/promotional webpage pattern with instructional tone throughout.<br><br>"If you have any questions about the product," sets up a conditional clause requiring a response action.<br><br>Final token "product," closes a conditional clause mid-thought, immediately expecting "please contact us" or similar follow-up directive. |
| λ=0.006 | 26 | 0.656 | Product description format<br><br>Customer service transition<br><br>"For further information," |
| λ=0.015 | 29 | 0.430 | product description<br>customer inquiry<br>"if you have any questions about the product," |
| λ=0.03 | 14 | 0.516 | For more details, |

### Sample 49
> … harmful bacteria. People with HIV/AIDS are more susceptible to infections.

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 133 | 0.556 | Enumerated list structure with bolded bullet points (e.g., "A weakened immune system") establishes a pattern of health-related risk factors. More risk factors are expected to follow this format.<br><br>Content domain signals: article systematically covers HIV/AIDS symptoms and risk factors, with the current section ("What Increases Your Risk for HIV Infection?") requiring specific medical risk factors for HIV transmission.<br><br>Final token ends mid-sentence after "HIV infection." — a period closes this introductory clause, signaling the next tokens should continue with another bolded bullet point or risk factor, parallel to "A weakened immune system." |
| control RL (λ=0) | 121 | 0.744 | Enumerated list structure: text has systematically covered HIV/STDs, heart disease, and cancer risks for people living with HIV; now completing the final section.<br><br>Medical/educational content pattern: each condition section presents a named health issue, explains its connection to HIV, and describes risk factors — the same format is expected here.<br><br>Final token "system." is a sentence opener mid-section, beginning a new sentence about oral health complications in people living with HIV; next tokens must continue this medical explanation, likely naming specific oral conditions or consequences. |
| λ=0.001 | 100 | 0.770 | Medical article structure: each disease section follows symptoms → transmission → treatments → complications pattern.<br><br>Content momentum: HIV section is mid-explanation, having stated "patients with HIV are more susceptible," requiring elaboration on specific oral health complications (e.g., gum disease, mouth infections).<br><br>Final token: "virus." ends a subordinate clause explaining HIV's effect; next tokens must continue describing how HIV compromises oral health, likely with specific examples. |
| λ=0.002 | 75 | 0.783 | Informational article structure covering oral health topics systematically, with each paragraph introducing a new subtopic.<br><br>Narrative momentum: HIV-related oral health risks being introduced, requiring elaboration on infection impacts.<br><br>Final sentence ends mid-thought after "patients are more susceptible to infections," requiring continuation explaining specific oral infections or complications. |
| λ=0.006 | 34 | 0.654 | Health article structure continuing<br><br>Explanation of HIV/AIDS impact on mouth<br><br>"AIDS weakens the immune system." |
| λ=0.015 | 31 | 0.150 | health article format<br>oral health conditions<br>"Individuals with HIV/AIDS are more susceptible" |
| λ=0.03 | 15 | 0.408 | Patients with HIV/AIDS. |

### Sample 50
> …Matthews took the field for individual drills. Matthews was acquired by the

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 108 | 0.631 | Biographical narrative structure: article profiles Schuler's career, moving chronologically from college to NFL, so his current team affiliation is expected next.<br><br>The sentence begun with "He joined the" promises completion with Schuler's current NFL team (Bengals or another organization), requiring proper noun identification.<br><br>Final token "the" is a definite article beginning a noun phrase, immediately requiring a specific NFL team name (e.g., "Bengals") as its head noun. |
| control RL (λ=0) | 115 | 0.820 | Sports news article reporting on Buffalo Bills' 2016 NFL draft, establishing factual journalistic tone throughout.<br><br>Narrative momentum: the article has listed two key free agent additions (Smith, Davenport), and is now introducing their role/significance, following the structure of announcing and explaining each player.<br><br>Final token "the" is a definite article beginning a noun phrase, requiring a specific team name (likely "Buffalo Bills") or organization following immediately, consistent with the article's focus on Bills roster moves. |
| λ=0.001 | 87 | 0.659 | Sports journalism narrative about NFL free agency, establishing background on Ryan's contract situation.<br><br>Continuation of Ryan's free-agent status explanation, requiring completion of who he was released from.<br><br>Final token "the" begins a noun phrase mid-sentence ("to the [team]"), immediately requiring a specific NFL team name (e.g., "the Giants" or "the Packers"). |
| λ=0.002 | 68 | 0.217 | Sports reporting context about a trade, requiring factual transaction details.<br><br>Narrative momentum: the trade just completed, now detailing consequences for the Bills.<br><br>"to the Bills" ends mid-phrase, immediately requiring the object — likely "Bills" or a new team's name. |
| λ=0.006 | 28 | 0.562 | Bengals trade context<br><br>contract/trade details<br><br>"to the" |
| λ=0.015 | 26 | 0.527 | NFL free agency<br>Bills roster<br>"signed to the" |
| λ=0.03 | 13 | 0.165 | and the |

### Sample 51
> …om its online stores – also continued to show healthy growth. Revenues were

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 159 | 0.637 | Financial data pattern: the text lists year-over-year revenue figures, e.g., "up 16 percent to $47.3 billion," "up 6 percent to $13.7 billion," establishing a repetition template of percentage increase followed by dollar amount.<br><br>Narrative momentum: the revenue section has been systematically listing business segments (enterprise, government, consumer electronics, mobile broadband, mobile services, mobile handsets); all segments have been covered, and the sentence structure signals the final figure is being provided.<br><br>Final token "were" opens a past-tense predicate completing the mobile handsets revenue figure clause, directly requiring a dollar amount and percentage increase (likely following the "up X percent" pattern used consistently throughout). |
| control RL (λ=0) | 132 | 0.614 | Financial news article reporting on Apple's quarterly earnings, maintaining consistent pattern of presenting revenue figures with context and comparison data.<br><br>The sentence structure "Revenue was" sets up an expected numeric figure (likely in millions/dollars), consistent with the article's pattern of reporting dollar amounts with precise figures (e.g., "$7.5 billion," "$12.6 billion").<br><br>Final token "were" is a passive past-tense verb beginning a revenue figure statement, requiring an immediate numeric value (e.g., "$X billion") to complete the clause, matching the article's established financial reporting style. |
| λ=0.001 | 103 | 0.728 | Financial news report tracking Microsoft's Azure business performance, with consistent pattern of reporting revenue figures for each cloud segment.<br><br>Narrative momentum: the article has been detailing Azure revenue breakdowns, and "revenues were" signals the next specific financial metric (likely $X billion) for the enterprise segment.<br><br>Final token "were" opens a passive construction mid-sentence, immediately requiring a dollar amount or numerical figure followed by a currency qualifier like "billions." |
| λ=0.002 | 67 | 0.685 | Financial report structure: revenue figures being presented sequentially across products.<br><br>Continuation of the "data centre, mobile" product breakdown pattern, each segment getting sales figures.<br><br>"revenues were" is an incomplete passive construction requiring a specific dollar amount or percentage figure to follow immediately. |
| λ=0.006 | 32 | 0.611 | financial reporting structure<br>revenue figures expected<br>"revenues were" requires a numeric value |
| λ=0.015 | 25 | 0.633 | financial reporting<br>revenue figures<br>"revenues were" |
| λ=0.03 | 14 | 0.598 | revenues were |

### Sample 52
> … hit 9 billion rubles, but the project stopped on the level of design works

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 128 | 0.626 | News report structure following inverted pyramid: headline summary done, now detailed factual elaboration expected, covering construction status, financial delays, or government response.<br><br>Chronological/progressive narrative momentum: the text traces a timeline ("since 2009") and "design works were terminated," implying a contrast between intended progress and actual status requiring elaboration.<br><br>Final token "works" ends mid-clarification after "design works were terminated" — a clause requiring grammatical completion, likely specifying the reason/when/how, e.g., "without completion" or "due to." |
| control RL (λ=0) | 127 | 0.742 | News article style reporting a Russian government decision, establishing factual/informational tone throughout.<br><br>The sentence ending "not proceeded beyond the design works" introduces a cause-effect explanation about the station's non-completion, strongly implying the reason or consequence will follow — likely citing funding, legal, or geopolitical issues.<br><br>The final phrase "design works" is the object of "proceeded beyond," completing a subordinate clause explaining the station's incomplete state; the main clause or next sentence must provide the cause, e.g., "due to insufficient funding" or "because of the conflict." |
| λ=0.001 | 95 | 0.737 | News article reporting on Russian-Macedonian airport deal, establishing factual context about project delays.<br><br>The phrase "did not go any further than design works" signals an explanation of why the project stalled, implying further elaboration on consequences or timelines follows.<br><br>Final token "works" closes a subordinate clause explaining the project's status; next tokens likely continue with a period and additional detail about what happened next or official statements. |
| λ=0.002 | 81 | 0.576 | News report structure: factual, third-person journalistic prose about stalled infrastructure project.<br><br>Narrative momentum: problem stated (delays), cause explained (design works halted), next expected detail about current status or consequences.<br><br>Final token "works" closes a clause explaining halt of design works; immediate next token likely continues with additional details or a contrasting complication. |
| λ=0.006 | 28 | 0.501 | news article format<br><br>project development context<br><br>"design works" ends sentence |
| λ=0.015 | 23 | 0.461 | construction news<br>contract details<br>"design works" |
| λ=0.03 | 13 | 0.164 | design works |

### Sample 53
> …caping the containment. Construction of Belarusian-1 and Belarusian-2, both

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 124 | 0.544 | Ongoing news article about Ukraine's nuclear power plant expansion, establishing factual/infomatic register throughout. The article is detailing VVER reactor specifics and construction details.<br><br>Narrative momentum: the article has introduced the 600MWe units and is now providing technical specifications, requiring reactor type (VVER) and capacity details to follow.<br><br>Final token "both" introduces a paired construction ("both [reactor type]"), requiring a noun complement immediately — almost certainly "VVER reactors" or similar, completing the parallel reference to the units just named. |
| control RL (λ=0) | 135 | 0.654 | Structured article format (headlines, summaries) signals factual, news-reporting style throughout — next text will continue this style with technical nuclear plant details.<br><br>The sentence "Both power stations, both" sets up an anaphoric repetition pattern, strongly implying parallel description of the two reactors — location, technical specs, or operational status.<br><br>Final token "both" is the second of two coordinating "both" constructions, requiring immediate continuation with a shared attribute describing the two VVER reactors in the Tashkent region (e.g., "of VVER-1000" or "built by..."). |
| λ=0.001 | 85 | 0.667 | News article structure: factual reporting on nuclear power plant closure, establishing context about site and reactors.<br><br>List/intro pattern: "two reactors, both" sets up a description of reactor type, location, or characteristics (e.g., VVER design, Russian-built).<br><br>Final token "both" opens a coordinated predicate requiring completion specifying shared attributes of the two reactors. |
| λ=0.002 | 61 | 0.571 | News wire format reporting a nuclear plant shutdown, establishing technical detail context.<br><br>Specific factual continuation expected: VVER reactor type and location must be provided.<br><br>"both" ends the sentence, requiring immediate specification of reactor type(s) and their characteristics. |
| λ=0.006 | 31 | 0.439 | Nuclear plant context established<br>technical specifications required<br>"both" requires paired reactor details |
| λ=0.015 | 26 | 0.266 | nuclear plant details<br>technical specifications<br>"Both, both" |
| λ=0.03 | 14 | 0.062 | Both, both |

### Sample 54
> …his case circa 1817). Hervey is also a thoughtful observer of what is going

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 127 | 0.572 | Historical narrative momentum: the scene sets up an investigation into a "great crisis" affecting the frontier, requiring its subject and cause to be named.<br><br>Literary/fictional register: prose novel describing character's thoughts ("what was going"), suggesting continuation in third-person limited or omniscient narrator style.<br><br>Final token "going" is part of the phrase "what was going [to happen/on]" — a standard construction requiring completion with an infinitive ("to happen") or continuation of the crisis subject (e.g., "to change," "on in the world"). |
| control RL (λ=0) | 141 | 0.571 | Narrative momentum establishing social/political tensions in pre-revolutionary Ireland — the "What is going on" phrase signals a broad social or political question needs completion.<br><br>The phrase "What is going" strongly anticipates the word "on" (idiomatically completing "going on"), continuing the thought about social unrest, political upheaval, or revolutionary fervor in rural Ireland.<br><br>Final token "going" is mid-idiom within a quoted dialogue/exploration, requiring immediate completion as "on" to form the common phrase "What is going on," then likely continuing the character's internal thought or dialogue about Ireland's revolutionary context. |
| λ=0.001 | 80 | 0.698 | Narrative momentum: describing historical setting of 19th-century European rural life and social conditions.<br><br>Thematic focus: contrast between the sisters' privileged position and the social/political tensions around them.<br><br>Final phrase "what is going" ends mid-clause, requiring completion describing societal change, revolution, or political unrest threatening the estate. |
| λ=0.002 | 69 | 0.617 | Narrative momentum building toward describing the women's curiosity about the world beyond their home.<br><br>Thematic focus on female perspectives and expanding consciousness throughout the passage.<br><br>Final word "going" ends an incomplete clause "what is going," requiring immediate completion like "on" or "on beyond." |
| λ=0.006 | 44 | 0.697 | narrative momentum about societal change<br><br>historical context of late 1800s Irish emigration<br><br>"what is going" requires completion |
| λ=0.015 | 26 | 0.654 | social commentary<br>narrative description<br>"what is going" |
| λ=0.03 | 14 | 0.513 | what is going |

### Sample 55
> … heading for the Canadian border is a CN locomotive not wearing CN colors. 

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 123 | 0.210 | Photo caption series pattern: short captions describe rail scenes with visual details, establishing rhythmic, evocative prose style throughout.<br><br>Narrative/visual continuity: each caption builds a distinct rail scene; the next entry likely describes another unique train, track, or station scenario, possibly referencing "Olive Branch" or a new location.<br><br>Final token "passing" closes a scene about a train passing a signal; the series suggests the next token begins a new caption, likely a location name, a subject noun, or a verb initiating another photographic moment. |
| control RL (λ=0) | 113 | 0.413 | Photo caption series pattern: each caption describes a specific railway scene with location and subject detail, suggesting another descriptive caption follows.<br><br>Narrative momentum of railway photography blog: captions alternate between locomotive types (Amtrak, Canadian, Amtrak), suggesting continuation with another rail scene or vehicle focus.<br><br>Final token "cars" ends a caption describing parked railcars, establishing a subject-locative image; next tokens will likely begin a new caption with "Another..." or a location descriptor continuing the blog's pattern. |
| λ=0.001 | 92 | 0.727 | Photograph caption series pattern: each line describes a distinct railroad image with brief commentary, continuing sequentially.<br><br>Narrative/visual content: each caption introduces a new scene — engine, tracks, yard — establishing momentum toward another rail-related image description.<br><br>Final token ends a caption describing a locomotive parked in a turnout; next token will begin another new caption line, likely with a scene or subject description. |
| λ=0.002 | 63 | 0.702 | Catalogue/photography series pattern: each entry is a new photo with description.<br><br>Continuation of observational railway imagery with consistent descriptive caption style.<br><br>Final token ends a sentence describing rail maintenance activity; next entry likely introduces another distinct railway scene or location. |
| λ=0.006 | 31 | 0.167 | Photo caption sequence pattern<br><br>railway photograph gallery context<br><br>"painted undercarriage" |
| λ=0.015 | 25 | 0.208 | rail photography captions<br>date/location context<br>"empty train" |
| λ=0.03 | 11 | 0.509 |  |

### Sample 56
> …ter high school it’s like all of a sudden they care about what you’re doing

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 134 | 0.146 | Argumentative momentum: essay builds a contrast between two parenting styles, establishing a framework of comparison for subsequent discussion of modern vs. traditional approaches.<br><br>Repetition/continuation pattern: the "then versus now" framing is explicitly set up; "then" and "now" are promised but not yet delivered, creating strong expectation of completing that dichotomy.<br><br>Final token "doing" is mid-clause within a subordinate "now" clause, requiring immediate syntactic completion — likely a verb phrase like "and you're allowed" or "and it's okay," consistent with the "not being monitored" theme. |
| control RL (λ=0) | 134 | 0.616 | Narrative/argumentative momentum: essay explains peer pressure origins in adolescence, building toward how it affects personal identity and social status.<br><br>Continuation of parallel structure: the sentence lists how peers influence the narrator ("what your life is like," "what you're doing"), implying additional examples or elaboration of this pattern.<br><br>Final token "doing" ends an incomplete clause mid-thought, part of a list ("what your life is like and what you're doing"), expecting continuation such as "and how happy you are" or a contrasting "but" pivot emphasizing peer pressure's negative effects on authentic self-expression. |
| λ=0.001 | 90 | 0.684 | Ongoing narrative argument about peer relationships evolving beyond envy to genuine curiosity and connection.<br><br>Continuation of a list/elaboration pattern: the text is mid-sentence explaining what friends are now interested in discovering about each other.<br><br>Final token "doing" ends an incomplete clause ("what you're doing"), requiring continuation such as "now" or a comma plus further elaboration on mutual curiosity. |
| λ=0.002 | 75 | 0.612 | Narrative momentum building toward a personal reflection on social comparison and self-doubt<br><br>List of social media pitfalls is being elaborated with specific examples and consequences<br><br>Final clause "what you're doing" ends mid-thought, expecting completion about how others' lives are judged or contrasted against one's own |
| λ=0.006 | 32 | 0.516 | blog post narrative voice<br><br>social comparison theme continuing<br><br>"what you're doing" requires completion |
| λ=0.015 | 26 | 0.454 | first-person narrative<br>relationship observation<br>"what you're doing" |
| λ=0.03 | 15 | 0.140 | what you're doing |

### Sample 57
> …egimen or from GVHD. Treatments focusing on easing the symptoms of oral muc

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 144 | 0.578 | Academic medical review article on chemotherapy toxicity, establishing pattern of defining then characterizing each toxicity category systematically.<br><br>The text is mid-sentence defining "mucositis/mucos" — clearly truncating "mucositis," requiring completion of this term before elaborating on its pathophysiology, risk factors, and management, consistent with the article's structure of explaining each toxicity type in depth.<br><br>Final token "muc" is a truncated mid-word continuation of "mucositis," requiring immediate completion as "ositis," then the sentence will continue describing mucositis etiology, severity, or clinical presentation as per the established pattern. |
| control RL (λ=0) | 135 | 0.513 | Medical review paper on mucositis treatment, establishing a systematic structure where each section covers a category of treatments with evidence-supported findings.<br><br>The text has followed a consistent pattern: introduce treatment category, list supportive evidence, cite studies, then summarize outcomes — this pattern strongly predicts continuation of the mucositis treatment discussion.<br><br>Final token "muc" is a truncated word mid-phrase ("oral mucositis"), part of a bullet point list of treatment types; the next tokens must complete "mucositis" before continuing with another bullet-pointed treatment category (e.g., "Oral mucositis"). |
| λ=0.001 | 96 | 0.608 | Medical/clinical content establishing evidence-based treatment guidelines for radiation therapy mucositis management.<br><br>Continuation of a structured list of "oral mucositis treatment options" with prior examples ("mouthwash, oral gel") setting expectation for another method.<br><br>Final token "oral muc" is the beginning of "oral mucositis," mid-word truncation, requiring completion as the subject of the next treatment option or section title. |
| λ=0.002 | 72 | 0.634 | Medical article following clinical content structure, moving through treatment sections systematically.<br><br>The section "Management of oral mucositis" is being introduced, signaling clinical guidance ahead.<br><br>"oral muc" is a truncated word, almost certainly completing as "mucositis," continuing the heading for the management section. |
| λ=0.006 | 31 | 0.360 | Medical clinical trial description<br><br>Symptom-focused enumeration<br><br>"oral muc" is partial word |
| λ=0.015 | 25 | 0.611 | clinical presentation<br>oral mucositis<br>"oral muc" |
| λ=0.03 | 13 | 0.260 | oral muc |

### Sample 58
> …oor: by me if any man enter in, he shall be saved, and shall go in and out,

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 141 | 0.233 | Biblical quote in progress: the model must complete the verse "enter in and go out, and feed your sheep" from John 10:9, continuing the parallel "enter in" structure.<br><br>Theological/expository context: the entire passage is a gospel commentary, building from a historical example (Hagar/Sarah) toward a spiritual parallel, expecting application of Hagar's symbolic meaning to the church.<br><br>Final token is "go out," the second parallel clause of the quoted verse; the immediate constraint is completing the biblical quote with "and feed your sheep" then likely explaining this verse's relevance to Hagar's symbolism. |
| control RL (λ=0) | 135 | 0.400 | Biblical/devotional register: theological exposition with scriptural citation ("Ephesians 2:17-18") establishing context for religious content.<br><br>Scriptural continuation momentum: the quotation is mid-sentence, requiring completion of Jesus' words about "going in and out freely" — a well-known passage from John 10:9.<br><br>Final token is a comma after "and go out," mid-quotation from the opened Bible verse, immediately constraining the next tokens to be the remaining text of the verse, likely "and feed your sheep" (John 10:9). |
| λ=0.001 | 116 | 0.510 | Religious/evangelistic content explaining church growth in Africa, establishing a contrast between African believers' spiritual experience and outsiders' perception.<br><br>The quote from Christ is mid-sentence, describing the privileges of sheep entering God's fold, requiring completion of the biblical phrase "and go in, and come out" referencing open doors.<br><br>Final token is a comma mid-clause within the Christ quote, immediately constraining continuation to complete the paired verb phrase "and go in, and come out" describing free access to God. |
| λ=0.002 | 77 | 0.418 | Biblical exposition structure: passage analyzed, now transitioning to supporting scripture.<br><br>Repetition/continuation of Jesus' words from John 10:9 about entering through Him.<br><br>Final token ends mid-sentence after "free to go in and out," requiring the closing words of that verse: "and find pasture." |
| λ=0.006 | 35 | 0.118 | Biblical scripture citation expected<br><br>Scripture completion required<br><br>"and go out," follows standard Gospel verse |
| λ=0.015 | 27 | 0.075 | Biblical quotation<br>freedom motif<br>"enter in,out" |
| λ=0.03 | 15 | -0.547 | and go in, |

### Sample 59
> …dule. The main thing we've got to finish is the northbound structure and we

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 103 | 0.416 | Direct speech continuation: a contractor's quote is mid-sentence, requiring completion of his ongoing thought about project status.<br><br>Narrative momentum: the sentence describes the project nearing completion, implying optimistic forward-looking language about finishing timelines or milestones.<br><br>Final token "we" is part of the first-person plural "we are going to have our final segment open before" structure, requiring a time/condition clause like "we complete construction" or a specific deadline phrase. |
| control RL (λ=0) | 123 | 0.766 | Direct quote in progress: an official's speech is being reported, requiring completion of the spoken statement's thought.<br><br>Sequential construction milestone narrative: "we have to do our east side, then the south side...and we" signals a list of remaining tasks still in progress, requiring continuation of the construction plan.<br><br>Final token "we" is the subject of an incomplete clause within the direct quote, immediately requiring a verb phrase (e.g., "are on schedule," "should be done by") describing the current status or next step of the construction work. |
| λ=0.001 | 77 | 0.754 | Direct quote from Jim Langer is ongoing, requiring continuation of his speech about project timeline.<br><br>The sentence structure "and we" sets up a paired completion about what remains to be done after the bridge work.<br><br>Final token "we" is mid-sentence within the quote, demanding immediate completion describing remaining project tasks or milestones. |
| λ=0.002 | 79 | 0.685 | Direct speech continuation: Bob's quote is mid-sentence, requiring completion of his thoughts.<br><br>Construction project context: the narrative follows a progress timeline with specific milestones mentioned.<br><br>Final token "we" begins a first-person plural verb phrase, most likely continuing with a future action like "should" or "expect" about the remaining phase. |
| λ=0.006 | 30 | 0.581 | Construction update narrative<br><br>Project completion timeline framing<br><br>"and we" opens coordinated clause |
| λ=0.015 | 23 | 0.590 | construction progress<br>schedule update<br>"and we" |
| λ=0.03 | 13 | 0.339 | and we |

### Sample 60
> … color making it the perfect addition to any front door. - 18 x 30 in - 100

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 131 | 0.693 | Product specification list format established: structured bullet-point attributes being enumerated for a product page, expecting material composition details next.<br><br>Material specification pattern: fabric composition percentage expected after "100" (mirroring earlier mention "100% polyester," strongly predicting "% polyester" follows immediately).<br><br>Final token "100" is the start of a material composition percentage figure mid-phrase, requiring "% polyester" to complete the standard fabric specification convention (e.g., "100% polyester" or "100% polyester/70% cotton/30% spandex"). |
| control RL (λ=0) | 128 | 0.728 | Product description format established: e-commerce listing style with fabric specs following bullet points, requiring continuation of technical specification details.<br><br>Percentage completion of a percentage composition — "100%" strongly signals a textile composition term like "100% cotton" or "100% polyester," typical of fabric specification language in product listings.<br><br>The final token "100" is the beginning of a percentage figure mid-specification, almost certainly completing as "100% cotton" or "100% organic cotton," requiring a "%" symbol and fabric name immediately next. |
| λ=0.001 | 93 | 0.715 | Product description genre with material/specs listing, following e-commerce bullet-point format.<br><br>Material composition details expected: "%100" signals a percentage composition phrase, e.g., "100% organic cotton" or "100% recycled polyester."<br><br>The final token "100" is the start of a percentage figure, immediately requiring "%" followed by material type to complete the specification. |
| λ=0.002 | 67 | 0.211 | Product specification format following structured material/quality bullet points.<br><br>List of percentage-based fabric composition details expected to continue.<br><br>"100%" is the start of a percentage specification, immediately requiring a fabric type or material name (e.g., "100% Cotton"). |
| λ=0.006 | 29 | 0.737 | Product description format<br>material composition expected<br>"100" begins percentage |
| λ=0.015 | 24 | 0.682 | product description<br>material composition<br>"100" |
| λ=0.03 | 14 | 0.161 | 100 |

### Sample 61
> …elow is a visual aid that shows the 3 (reference) axes of rotation. This is

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 119 | 0.799 | Forum reply responding to a request for a "visual of pitch angles on a missile," establishing context of a military/munitions forum post.<br><br>The author has been building toward providing a specific reference or image, having introduced "the diagram shown below" with a link, suggesting they will now describe or contextualize the diagram visually.<br><br>"This is" ends the sequence — an introductory declarative sentence requiring a noun phrase or predicate describing the diagram's nature, purpose, or specific contents (e.g., "a cross-sectional view of..."). |
| control RL (λ=0) | 140 | 0.811 | Technical/educational aviation discussion explaining aircraft reference frames (earth-fixed vs. aircraft-fixed) in context of torque vectoring and aerodynamics.<br><br>The conversation is building toward defining a specific technical diagram or coordinate system (likely Euler angles, yaw/pitch/roll axes, or aircraft frame diagram), requiring an explanatory description.<br><br>"This is" is an incomplete predicate at the very end, functioning as a demonstrative/definitional clause; it immediately constrains the next token to be a noun or noun phrase identifying what the diagram/figure represents (e.g., "the aircraft's coordinate system," "a standard yaw-pitch-roll diagram"). |
| λ=0.001 | 96 | 0.756 | Technical explanation of aviation mechanics with consistent diagrammatic/descriptive illustrations throughout.<br><br>The phrase "This is" sets up an explanatory clause describing the figure or concept just shown, requiring a descriptive noun phrase or elaborating statement.<br><br>Final token "is" opens a predicate completing "This is," demanding an immediate noun phrase or qualifier describing the preceding visual concept (e.g., "a diagram," "the aircraft's..."). |
| λ=0.002 | 61 | 0.775 | Scientific explanation of three-axis coordinate system with consistent diagram-description pattern.<br><br>Continuation of a detailed description of a visual diagram or projection setup.<br><br>"This is" ends an incomplete sentence introducing what the coordinate system/diagram represents or illustrates next. |
| λ=0.006 | 25 | 0.672 | Technical aerospace explanation<br>Reference to coordinate system<br>"This is" |
| λ=0.015 | 22 | 0.665 | technical diagram<br>coordinate system<br>"This is" |
| λ=0.03 | 12 | 0.527 | This is |

### Sample 62
> …. But be forewarned!! Some quiet afternoon while watching TV, you, too, may

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 160 | 0.677 | Narrative setup establishing a relatable "we may all do this" scenario, inviting readers to connect with the author's experience. The second-person address creates a direct community appeal, suggesting the next text continues building this personal anecdote into a cautionary or comedic point.<br><br>The contrast between the mundane (snacking, watching TV) and the absurd (alien invasion) builds comedic tension, priming the reader for an unexpected punchline or twist — likely revealing the narrator is imagining something absurd or funny.<br><br>"may" ends a "you too" rhetorical setup mid-sentence, immediately requiring a verb phrase completing the parallel condition (e.g., "may find yourself..."), continuing the self-deprecating, humorous tone of the anecdote. |
| control RL (λ=0) | 135 | 0.685 | Humorous/delightful narrative tone established throughout, building toward a funny or amusing observation about viewers reacting to the film's content (the film is about a woman in a wheelchair who finds humor in mundane life).<br><br>The rhetorical structure "Just because you saw this movie on TV... you too, may" sets up a parallel to the narrator's own experience, implying the reader will experience similar laughter or distraction.<br><br>Final token "may" opens an infinitive clause, requiring a verb phrase describing a likely human reaction, e.g., "find yourself laughing uncontrollably" or "suddenly relate to her perspective." |
| λ=0.001 | 91 | 0.480 | Humorous/ironic setup establishing a relatable scenario about mind-wandering, requiring a punchline payoff.<br><br>List of "what you may" implies continuation of hypothetical consequences or outcomes of daydreaming.<br><br>Final token "may" opens a modal verb construction, immediately requiring a verb phrase like "find yourself" or "wander into" completing the ironic parallel with the show's premise. |
| λ=0.002 | 76 | 0.570 | Humorous, conversational tone established throughout — light, playful humor expected to continue.<br><br>Narrative momentum building toward a relatable personal anecdote about a surprising "Ah Ha!" moment.<br><br>"may" ends the sentence mid-clause, requiring a modal verb completion like "may find yourself" or "may think." |
| λ=0.006 | 31 | 0.658 | Narrative setup completed<br>humorous domestic scenario<br>"you too may" requires completion |
| λ=0.015 | 29 | 0.599 | TV show description<br>suspenseful narrative<br>"you too, may" |
| λ=0.03 | 13 | 0.367 | you may |

### Sample 63
> …wn at artropolis: “bright side” up close: another fave on diplay at roy boy

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 130 | 0.352 | Biographical profile format for a photography award winner, following a structured pattern of credentials, achievements, and institutional affiliations.<br><br>The sentence ending mid-flow lists "Renee Rappole and Renee Boy" is introducing co-winners (likely co-photographers or co-submitters) with "and," so a surname completion and likely a second person or institutional credit follows.<br><br>Final token "Boy" is the last syllable of a surname, mid-phrase after "and," immediately requiring surname completion (additional letters) and likely a title/role for this co-entrant. |
| control RL (λ=0) | 112 | 0.371 | Artist biography following standard museum/catalog format, moving through exhibitions, institutions, and collaborators sequentially.<br><br>The text is mid-sentence describing a specific exhibition with collaborators "russell boy" — a truncated proper name (likely "Russell Boyle" or similar photographer) requiring completion.<br><br>Final token "boy" is the incomplete last name of a collaborator in "russell boy[le/xx]," immediately constraining the next tokens to complete that surname before continuing with exhibition details or publication credit. |
| λ=0.001 | 89 | 0.266 | Credit/attribution continuation: the text is mid-sentence crediting "Randy Ray & Randy Boy" as collaborators.<br><br>Press release format: institutional announcement tone signals formal acknowledgment of production collaborators.<br><br>Final token "boy" is part of the collaborator name "Randy Boy" (likely "Randy Boyce" or similar), requiring completion of that name before further credit details. |
| λ=0.002 | 64 | 0.418 | Press release format following structured announcement style throughout.<br><br>The sentence lists photographer credits, establishing a series pattern requiring completion.<br><br>"robert boy" is a truncated name mid-word, strongly constraining next tokens to "d" followed by continuation of the production credit. |
| λ=0.006 | 29 | -0.666 | artist biography format<br><br>credits/attribution pattern<br><br>"roy roy" likely continues |
| λ=0.015 | 24 | 0.312 | artist credits<br>production credits<br>"robert boy" |
| λ=0.03 | 13 | -1.133 | ro boy |

### Sample 64
> …00,000-run edition came out shortly before MBS embarked on a publicity tour

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 109 | 0.613 | Political/religion news article style, covering Catholic-Protestant tensions, with formal journalistic register throughout.<br><br>Narrative momentum building toward explaining what Trump was doing during the papal visit that caused controversy — a direct explanation is expected after "that led to the controversy."<br><br>Final token "tour" is mid-phrase, part of "a planned global tour," requiring immediate continuation specifying the destination(s) or context of that tour, likely referencing the U.S. or a specific location. |
| control RL (λ=0) | 124 | 0.648 | Narrative momentum: the sentence about the "grand tour" is mid-clause, requiring completion of what the tour involved or where it took place.<br><br>Thematic pattern: article chronologically traces Obama's foreign policy, so the grand tour detail will likely reference specific destinations like Asia/Africa engagements.<br><br>Final token "tour" ends an incomplete subordinate clause ("in connection with a grand tour"), grammatically and semantically requiring a prepositional phrase or relative clause specifying where/when this tour occurred (e.g., "of Asia" or "in Asia"). |
| λ=0.001 | 89 | 0.635 | News article structure: after background on Trump-Pence friendship, narrative shifts to Trump's political behavior requiring elaboration.<br><br>Causal/explanatory momentum: the text establishes a chain explaining Trump's actions; Pence's visit is the latest clue to be developed.<br><br>Final token "tour" ends an incomplete clause describing a "political goodwill tour," requiring immediate continuation specifying its destination or purpose. |
| λ=0.002 | 75 | 0.552 | Narrative momentum building toward describing Trump's "tour" of foreign capitals, establishing context for the Saudi/Palestine issue.<br><br>Chronological political journalism structure moving backward from present event to preceding actions.<br><br>Final token "tour" ends an incomplete clause describing Trump's foreign trip, requiring continuation specifying destinations or purpose. |
| λ=0.006 | 44 | 0.589 | political investigative journalism<br><br>narrative momentum toward Obama's actions<br><br>"took him on a goodwill tour" requires completion describing the tour's destination or purpose |
| λ=0.015 | 25 | 0.472 | political scandal<br>Senate hearings<br>"congressional tour" |
| λ=0.03 | 13 | 0.029 | tour tour |

### Sample 65
> …o, MSN, along with Bing Besides using absolutely free programs like these. 

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 116 | 0.750 | Listicle/SEO-style blog content pattern: fragmented, loosely connected paragraphs with bullet-point and numbered list structures repeating throughout.<br><br>Continuation of "free dating sites for seniors" topic: each paragraph introduces a new angle or feature of online dating for older adults, suggesting another list item or recommendation follows.<br><br>Final sentence introduces "various other features" as a new list point, implying an enumerated series is beginning — the next tokens will likely be a dash, bullet, or number introducing specific features/services available on dating sites. |
| control RL (λ=0) | 117 | 0.733 | Promotional/marketing register with repetitive structure: each paragraph introduces an online dating platform with benefits and features, establishing a listicle pattern of platform descriptions.<br><br>Semantic expectation of continued platform promotion: the text consistently pairs "some other [platform] might have [features]" with positive claims about functionality and accessibility, predicting similar elaboration.<br><br>Final sentence ends mid-promotional list with "Also" — a coordinating additive conjunction signaling continuation of the enumeration of dating platform features or considerations, likely introducing another benefit, alternative, or concluding statement. |
| λ=0.001 | 85 | 0.756 | Continuation of a listicle/FAQ about free dating site alternatives, suggesting more comparative points or examples will follow.<br><br>The text contrasts free vs. paid services, implying further elaboration on benefits, drawbacks, or recommendations is expected.<br><br>Final sentence ends mid-thought comparing free site providers; next tokens likely continue elaborating on options, features, or use cases. |
| λ=0.002 | 79 | 0.763 | Promotional/informational web content about online dating apps, maintaining a structured listicle format with bullet points.<br><br>Continuation of app-recommendation section with a pattern of feature highlights and user benefits.<br><br>The final sentence ends a paragraph about Facebook and Google+, signaling the next section will begin a new bullet-pointed feature or recommendation. |
| λ=0.006 | 32 | 0.734 | List/enumeration pattern established<br><br>Online dating app content<br><br>Final sentence introduces additional app types |
| λ=0.015 | 26 | 0.522 | list continuation<br>advice format<br>"Also you can use..." |
| λ=0.03 | 11 | 0.624 |  |

### Sample 66
> …, all of Clovis, and James Olguin of Mesa, Ariz.; four daughters, Martha "M

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 160 | — | _(extraction failed)_ |
| control RL (λ=0) | 117 | 0.388 | Ongoing quote from Sharon Lutz, the author of "The Road to Recovery," mid-sentence requiring completion of "M[ary/Mary Anne]"<br><br>The list of community members is being enumerated, establishing a pattern of proper names with titles/roles ("Dale K. Lutz...Mary")<br><br>Final token "M" is the beginning of a truncated proper name within an em-dash-quoted title list, immediately requiring the rest of the name (e.g., "ary Anne") to complete the citation. |
| λ=0.001 | 92 | 0.411 | Obituary/tribute genre with personal anecdotes and community recognition, continuing to list family members.<br><br>The phrase "such as M" signals an incomplete list of names, requiring completion of "M" with a given name.<br><br>Final token "M" is the beginning of a proper name (likely "Marilyn" or "Marian"), part of a family tribute clause introducing more named relatives. |
| λ=0.002 | 68 | 0.252 | Bulleted list of named recipients continuing a consistent naming pattern.<br><br>Each recipient follows "Name: Title" format, with a colon after surname.<br><br>The final token "M" begins a truncated surname for "Marilyn" completing "Marilyn: [Title]." |
| λ=0.006 | 32 | 0.089 | Local news narrative continues<br><br>charity event details<br><br>"M" is the start of a name |
| λ=0.015 | 26 | -0.025 | obituary format<br>family members listed<br>"MomM" |
| λ=0.03 | 11 | -0.071 | M |

### Sample 67
> …ur daily life. Learn more about each candidate to see if you feel they will

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 117 | 0.573 | Direct quote continuation: a partial quote opened with quotation marks and the word "will" requires completion, almost certainly "best represent" or similar phrasing.<br><br>Candidate questionnaire format: the text presents a structured list of questions directed at elected officials, with consistent phrasing and a closing question about vote confidence.<br><br>Final token "will" opens a verb phrase within a closing direct quote, immediately constraining the next words to a verb like "best," "most effectively," or "serve," completing the trust/vote confidence question. |
| control RL (λ=0) | 124 | 0.692 | Political candidate election guide genre, establishing criteria for voter decision-making ("how to choose," "how well they will").<br><br>The text is building toward a closing question to voters, asking them to evaluate candidates based on previously listed qualities (fiscal responsibility, economic growth, education, infrastructure, healthcare, environment).<br><br>Final token "will" opens a verb phrase completing the rhetorical question "which candidate do you believe will..." — next tokens must be a verb describing a candidate's future action/quality, most likely "best represent," "succeed," or "serve." |
| λ=0.001 | 99 | 0.475 | Direct voter appeal establishing criteria for candidate assessment, requiring completion of the conditional ("candidates who will...").<br><br>Parallel rhetorical structure: the text contrasts "will work for me" with "I will be able to make them work for me," building toward candidate qualification.<br><br>Final phrase "will" opens a verb clause completing the conditional about selecting candidates who align with the voter's interests, constraining next tokens to a future-tense action phrase. |
| λ=0.002 | 64 | 0.679 | Election voter guide format establishing candidate comparison structure.<br><br>Logical argument building toward candidate suitability, requiring a concluding assessment.<br><br>Final fragment "who do you think will" demands completion with a verb phrase like "best represent your values" or "make your town better." |
| λ=0.006 | 29 | 0.405 | Voter questionnaire format<br>Candidate evaluation questions<br>"that will" requires completion |
| λ=0.015 | 26 | 0.550 | election context<br>candidate evaluation<br>"do you believe they will" |
| λ=0.03 | 13 | 0.139 | will they |

### Sample 68
> …Fleming’s Triangle (Pickering’s Triangle). Full resolution: http://www.cele

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 112 | 0.467 | Blog/magazine post about telescope design and optics, establishing domain around DIY telescope building with a specific example lens configuration.<br><br>Narrative momentum: the text has introduced the lens design context and is transitioning to the author's practical demonstration, signaling an upcoming personal account or hands-on experiment with these optics.<br><br>Final token "cele" is the beginning of an incomplete word mid-sentence — almost certainly "celestial" or "celebrate," part of a sentence describing the telescope's subject matter or purpose. |
| control RL (λ=0) | 111 | 0.738 | Blog/membership post format: introduction to a website, followed by a membership request form, signaling a call-to-action continuation.<br><br>The membership form fields have been listed with labels, and "Website" is being completed mid-entry — a truncated URL or domain name is expected next.<br><br>"cele" is the beginning of a URL/website name (likely "celestial" or "celestialobjects.org"), mid-word, requiring immediate completion of that specific site name before the form fields conclude. |
| λ=0.001 | 83 | 0.692 | Blog post footer attribution pattern: author name begins after website description.<br><br>Acronym completion required: "cele" is the start of "celestron" or "celestronomer," fitting the astronomy enthusiast context.<br><br>Final token "cele" is a truncated word mid-attribution, immediately constraining next tokens to complete the author's name or title. |
| λ=0.002 | 73 | 0.598 | Copyright/attribution notice pattern: "©Cele" is the beginning of a standard photographer credit line.<br><br>Blog post metadata block signals a closing attribution field.<br><br>Final token "Cele" is a truncated proper noun (likely "Celestron" or the author's name), requiring immediate continuation of that word. |
| λ=0.006 | 32 | 0.672 | Website attribution<br>celestial photography site<br>"cele" is "celestron" abbreviation |
| λ=0.015 | 23 | 0.448 | website attribution<br>astronomy community<br>"cele" |
| λ=0.03 | 12 | -0.474 | cele |

### Sample 69
> …ng appearance. Once an order is placed, we ask that all changes (concerning

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 131 | 0.520 | Product listing page format with structured sections (Product Specs, Dimensions, Ordering, Notes), following retail/boating industry conventions. The "Notes" section signals a customer service/caveat notice.<br><br>A list of conditions or constraints for ordering is being enumerated, with "regarding" beginning a clause that typically follows in retail contexts like "regarding color," "regarding customizations," or "regarding special order requests."<br><br>Final token: "regarding" is a preposition introducing a noun phrase or relative clause, requiring a specific condition or topic the retailer wishes to communicate to buyers. |
| control RL (λ=0) | 122 | 0.634 | E-commerce product listing pattern with standard boilerplate sections: description, availability, warranty, and customization notice.<br><br>The phrase "changes concerning" is part of a standard legal/corporate disclaimer about custom orders, requiring completion with what the changes apply to — likely "color, sizing, material" or similar product attributes.<br><br>Final token "concerning" is a preposition beginning a prepositional phrase, grammatically requiring a noun phrase immediately following, almost certainly listing specific product customization attributes (e.g., "color, size, or material"). |
| λ=0.001 | 75 | 0.672 | E-commerce product page structure: specifications section transitioning to a returns/notes clause.<br><br>Legal/disclaimer tone expected: terms conditions language about order modifications or fabric substitutions.<br><br>Final token "concerning" opens a relative clause requiring specification of what order concerns apply, likely "fabric," "color," or "sizes." |
| λ=0.002 | 76 | 0.654 | Product listing details pattern: specifications, materials, and warranty information are being enumerated.<br><br>Formal e-commerce product description tone maintained throughout, with legal/disclaimer language.<br><br>Final fragment "concerning" is a preposition beginning a list of order customization conditions, expecting specific examples like color, size, or material choices. |
| λ=0.006 | 35 | 0.574 | Custom woodshop policy statement<br><br>Refund/return policy explanation<br><br>"concerning" introduces what qualifies |
| λ=0.015 | 27 | 0.475 | custom wood order<br>retail returns policy<br>"concerning" |
| λ=0.03 | 13 | 0.065 | regarding |

### Sample 70
> …y strikes at around 9-12 months, when sickness occurs, plus general changes

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 139 | 0.722 | Numbered list structure with consistent pattern: each point is a brief tip about dog potty training, implying continuation of the series.<br><br>The phrase "and changes" ends the final bullet point, strongly suggesting either a closing clause (e.g., "and changes in routine/diet") completing the thought, or a new list item continuing the potty training advice sequence.<br><br>Final token "changes" is the end of an incomplete clause mid-numbered-list-item, syntactically requiring a noun/phrase continuation (e.g., "in routine," "to the household"), or a new list item number like "6." to follow. |
| control RL (λ=0) | 125 | 0.693 | Continuation of a structured list of situations that trigger separation anxiety in dogs — "big changes" is the final item and more examples are expected (e.g., travel, moving house).<br><br>Domain/genre signals: pet care/welfare informational article with a Q&A format, providing practical advice to dog owners about anxiety.<br><br>Final token "changes" ends an incomplete list ("moving house or major life changes") — the sentence is grammatically open, requiring a closing phrase like "in the family/dog's environment" or a period/conjunction to close the thought. |
| λ=0.001 | 82 | 0.724 | Continuation of a list of stress triggers (vaccination, vet visits, new pets, new humans) requiring completion.<br><br>Educational/wellness advice tone maintained throughout, guiding pet owners toward practical, reassuring guidance.<br><br>Final phrase "major changes" is mid-list, strongly implying additional examples or a concluding clause explaining why these changes are stressful. |
| λ=0.002 | 59 | 0.733 | Product review/article format continues explaining factors affecting puppy sleep patterns.<br><br>List-style enumeration of sleep disruptors building toward a comprehensive conclusion.<br><br>"and other changes" is mid-list, strongly expecting continuation with specific examples like environments or household routines. |
| λ=0.006 | 33 | 0.429 | Pet grooming article context<br><br>symptoms/behaviors list<br><br>"changes" ends mid-phrase |
| λ=0.015 | 25 | 0.578 | dog training tips<br>stress causes<br>"and other changes" |
| λ=0.03 | 14 | 0.214 | and big changes |

### Sample 71
> …of the 19th ... Gold Mining Crushing Equipment Because gold mining plant is

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 115 | 0.648 | Search engine results/web scraping content about granite processing equipment, establishing a catalog/product description pattern. The text alternates between equipment listings and descriptive product content.<br><br>The pattern shifts into a descriptive sentence about "stone crushing plant," suggesting an explanatory definition or use-case description follows.<br><br>Final token: "is" is a linking verb mid-sentence, requiring an immediate complement—likely a noun phrase or adjective describing what a stone crushing plant is, e.g., "a key piece of equipment" or "used for crushing." |
| control RL (λ=0) | 132 | 0.730 | Search results page listing mining equipment and crusher articles, establishing a repetitive pattern of product descriptions, company names, and technical specifications.<br><br>The phrase "stone crushing plant is" opens a definitional/technical sentence, following the pattern of the earlier sentence "Crushing plant is one of the most important and common production lines." A similar technical definition or description of the plant's role/process is expected next.<br><br>Final token "is" is the predicate of an incomplete sentence requiring a noun phrase or adjective describing what a stone crushing plant is — likely "a key," "the core," or "used for." |
| λ=0.001 | 96 | 0.714 | Continuation of a structured informational page about stone crushing machinery, maintaining explanatory/educational tone throughout.<br><br>Argumentative momentum: "stone crushing plant is" sets up a definitional or descriptive clause elaborating on what it is or does.<br><br>Final token "is" opens a predicate requiring a complement — next tokens must define or describe stone crushing plant characteristics/function, e.g., "a key..." or "used for..." |
| λ=0.002 | 78 | 0.569 | Structured list of crushing plant FAQs, each with heading followed by explanatory prose.<br><br>Continuation of a new FAQ entry beginning with "What is a stone crushing plant is" — establishing definitional content expected.<br><br>Final token "is" begins a predicate clause defining what a stone crushing plant is, requiring a noun phrase or adjective continuation. |
| λ=0.006 | 27 | 0.375 | technical description pattern<br><br>product specification context<br><br>"is" begins a predicate |
| λ=0.015 | 25 | 0.665 | equipment description<br>process explanation<br>"stone crushing plant is" |
| λ=0.03 | 15 | 0.615 | stone crushing plant is |

### Sample 72
> …Being that we are a Painting company in Fort Lauderdale FL. We offer over 3

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 129 | 0.402 | Bulleted list of qualifications/attributes for a lawn care service, each item emphasizing experience and reliability (e.g., "25 years of lawn care," "3+ years").<br><br>Marketing/promotional copy pattern established, with benefit-focused language like "reliable," "professional" and "affordable" — next tokens continue this pattern.<br><br>The final token "3" is mid-sentence, beginning a specific number quantifier (e.g., "30 years of experience" or "3+ years of experience"), requiring completion of a quantitative claim about the company's tenure. |
| control RL (λ=0) | 134 | 0.518 | Promotional/marketing copy for a residential lawn care service, establishing credibility and professionalism throughout.<br><br>Numerical pattern continuation: "over 3" strongly implies a large number following (e.g., "over 30 years of experience"), a common industry credential phrase in landscaping/service business marketing.<br><br>Final token "3" is the beginning of a multi-digit age/experience figure within the phrase "we've been doing it for over 3___ years," requiring completion of a large number, almost certainly "30," "35," or "38" — a round figure typical in business descriptions. |
| λ=0.001 | 90 | 0.490 | Promotional/business description pattern established, emphasizing decades of experience and expertise.<br><br>The phrase "over 3" strongly signals a number continuation — completing the age/experience figure, e.g., "35 years of experience."<br><br>Final token "3" is the start of an age figure ("over 35 years"), requiring immediate numeric continuation like "5 years" or "0 years." |
| λ=0.002 | 73 | 0.508 | Service marketing page building credibility through experience and qualifications.<br><br>Numerical pattern continuation: "over 3" strongly implies a large number like "30" years or "35 years."<br><br>Final token "3" begins a specific numeric year count, constraining next tokens to digits completing a decade figure. |
| λ=0.006 | 28 | 0.339 | Business description pattern<br><br>experience credential expected<br><br>"3" begins a number |
| λ=0.015 | 25 | 0.393 | marketing copy<br>experience claims<br>"has over 3" |
| λ=0.03 | 15 | 0.292 | with over 3 |

### Sample 73
> … Combining massage, aromatherapy, acupressure, chiropractic treatments, and

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 109 | 0.809 | Promotional/commercial copy for a spa or wellness clinic, using persuasive, benefit-driven language throughout.<br><br>List-continuation pattern: "massage therapy, aromatherapy, guided visualization, craniosacral therapy, acupressure, and" signals an incomplete enumerated list requiring more items to complete it.<br><br>Final token "and" is a list conjunction mid-enumeration, requiring at least one more item (e.g., another therapy) to grammatically complete the series. |
| control RL (λ=0) | 119 | 0.786 | Enumerated list structure: "massage therapy," "guided meditation," "acupuncture," "chiropractic adjustments," and — a series of complementary holistic treatments still being listed.<br><br>Health/wellness promotional register: the text consistently uses professional, reassuring language describing alternative medicine benefits for migraine sufferers.<br><br>Final token "and" is a coordinating conjunction mid-list, requiring one or more additional holistic treatment items (e.g., "heat therapy," "dietary counseling") to complete the enumeration, likely followed by a closing phrase about these treatments. |
| λ=0.001 | 81 | 0.792 | Continuation of a list of holistic therapy techniques already introduced (craniosacral therapy, aromatherapy, massage), expecting more examples.<br><br>Marketing/promotional register for a wellness practice, maintaining reassuring, inclusive tone throughout.<br><br>Final token "and" is a conjunction mid-list, requiring at least one more complementary therapy modality to follow immediately. |
| λ=0.002 | 78 | 0.772 | Continuation of a list of treatment modalities in a holistic medicine marketing context.<br><br>The enumeration pattern ("massage therapy, cranial sacral therapy, and") requires at least one more item to complete the series.<br><br>Final token "and" is a list coordinator demanding the immediate next treatment modality, likely another alternative therapy item. |
| λ=0.006 | 29 | 0.687 | wellness treatment list<br>complementary therapies expected<br>"and" continues list |
| λ=0.015 | 25 | 0.638 | therapeutic methods<br>relaxation techniques<br>"and" |
| λ=0.03 | 12 | 0.353 | and |

### Sample 74
> …shen: 84.7 101. Village of Monroe: 81.06 145. City of Middletown: 72.68 147

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 136 | 0.655 | Rankings list pattern: numbered entries with city names, counties, and ratings (e.g., "86. New Orleans...87...88..."), requiring next rank "147" and its corresponding city/counties to follow.<br><br>Structured data format: each entry follows "rank. City, State...County...Rating" pattern, establishing rigid templating constraints for continuation.<br><br>"147" is an incomplete rank number in the established sequence, immediately requiring a period and city name (likely "New Orleans" again, consistent with prior rankings), followed by state, comma, and county names. |
| control RL (λ=0) | 138 | 0.711 | Structured list pattern: ranked cities with numerical positions (1, 2, 7, 10, 116, 147) followed by city names and rankings, establishing strong continuation expectations.<br><br>Ranking list momentum: the enumeration of ranked cities is incomplete; each entry follows "[number]. [City] No. [rank]" format seen repeatedly throughout.<br><br>Final token "147" is the start of the next ranked list item (mirroring "146. San Francisco"), immediately requiring a period "." followed by a city name and "No. [rank]" to complete the pattern. |
| λ=0.001 | 118 | 0.610 | Ranking list pattern: ranks 1-147 being listed sequentially with state rankings and scores, each entry following "[rank] [rank position] [score] [state] [score] [state]".<br><br>Incomplete sequence reaching rank 147 implies more rankings may follow or the list concludes.<br><br>Final token "147" begins a new rank entry in the list, immediately requiring a " [rank]" or "- [score] [state] [score] [state]" continuation matching the established format. |
| λ=0.002 | 101 | 0.669 | Structured ranking table with sequential position entries, each following the pattern "[Rank] [Ranking name], [state], [value]"<br><br>The list is mid-sequence, enumerating 2014 rankings by rank position<br><br>Final token "147" is a rank number mid-entry, requiring an immediate ranking name and state (e.g., "147. [Ranking Name], [state] - [value]") |
| λ=0.006 | 33 | 0.202 | Ranking list continuation<br>State-by-state performance data<br>147th position requires next entry |
| λ=0.015 | 28 | 0.191 | ranked list of cities<br>housing affordability index<br>"147" |
| λ=0.03 | 14 | 0.163 | 147 |

### Sample 75
> … Golden Globe for Best Actor. Murray can next be seen reprising his role in

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 114 | 0.641 | Narrative momentum: the text is listing cast member details and career highlights, establishing context for a specific announcement or role.<br><br>Repetition/continuation pattern: the phrase "The actor will appear in" mirrors earlier mentions of cast roles, strongly predicting a specific film title follows.<br><br>Final token "in" is a preposition beginning a noun phrase, requiring an immediate film title or production name as its direct object — likely a specific movie title (e.g., "The Muppets" or another project). |
| control RL (λ=0) | 112 | 0.693 | Celebrity gossip/entertainment news register, listing Scarlett Johansson's upcoming film projects with a pattern of "she will appear in [title]."<br><br>The list pattern established with "She will also appear in" signals at least one more movie title is expected to complete the enumeration.<br><br>Final token "in" is a preposition beginning a prepositional phrase that directly requires the title of another upcoming film Scarlett Johansson is starring in, as the grammatical and narrative completion of the sentence structure. |
| λ=0.001 | 93 | 0.664 | News article summarizing multiple cast-related stories about the "Ghostbusters" movie, establishing a pattern of introducing and elaborating on each cast member's activity.<br><br>Continuation of list/overview structure: two stories covered, now returning to the main subject for further development.<br><br>Final token "in" completes a prepositional phrase ("set to appear in"), immediately requiring the specific movie title or release context. |
| λ=0.002 | 64 | 0.650 | Article listing upcoming sequel projects for Ghostbusters 2016 cast members.<br><br>Continuation pattern: each cast member gets a brief project description.<br><br>Final token "in" begins a prepositional phrase specifying which sequel project Chris Hemsworth appears in next. |
| λ=0.006 | 32 | 0.559 | news article format continues<br><br>casting details expected<br><br>"recently in" requires a project name |
| λ=0.015 | 31 | 0.692 | movie sequel news<br>Evan Almighty context<br>"set to reprise his role in" |
| λ=0.03 | 14 | 0.599 | will star in |

### Sample 76
> …arp spines on the fore-shin can be drawn across a set of blunt processes on

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 159 | 0.533 | Scientific anatomical description of *Chelone* (a type of fossil nautilid), requiring precise terminology about shell structure — "fins" and "tubercles" suggest continued anatomical labeling.<br><br>The text is mid-sentence describing a structural feature of the shell's internal anatomy, expecting continuation of morphological description (e.g., "on the inner whorl" or "on the umbilical plate").<br><br>Final token "on" begins a prepositional phrase describing the surface/area of the tubercles, requiring an anatomical noun immediately following, likely a specific shell part (e.g., "the inner whorl," "the umbilicus," or "the periphery"). |
| control RL (λ=0) | 142 | 0.547 | Scientific anatomical description pattern: formal taxonomic/natural history prose detailing morphological features of a mollusk species.<br><br>Continuation of an anatomical feature list: the text is mid-sentence describing the radula's morphology, with "a series of teeth on the outer side of the radular cone" establishing a pattern of listing sequential anatomical structures.<br><br>Final token "on" is a preposition beginning a prepositional phrase, requiring an immediate anatomical location noun phrase — likely referencing parts of the proboscis or inner radular surface (e.g., "on the inner side" or "on the inner radular cone"). |
| λ=0.001 | 101 | 0.323 | Comparative anatomical description of chimaeroid fins mid-sentence, requiring continuation of anatomical detail.<br><br>Structural contrast pattern: "hard edges on one side... softer ones on" implies a paired anatomical comparison is being completed.<br><br>Final phrase "softer ones on" is mid-construction, requiring a specific anatomical location (e.g., "the other side" or a specific skeletal element) to complete the comparative description. |
| λ=0.002 | 89 | 0.501 | Scientific description of fish anatomy, building detailed morphological account of *Pterolepis*<br><br>Structural continuation of a paired anatomical feature: "paired keels on one side and a keel on the other"<br><br>Final phrase "a keel on" is mid-sentence, requiring anatomical location completion, likely "the pectoral fin" or similar body part |
| λ=0.006 | 32 | 0.394 | descriptive biological text<br>anatomical comparison structure<br>"on" requires anatomical noun |
| λ=0.015 | 26 | 0.473 | scientific description<br>anatomical structures<br>"small teeth on" |
| λ=0.03 | 14 | 0.066 | the base on |

### Sample 77
> …e the wonders as they occur around you. Don't claim them. Feel the artistry

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 129 | 0.533 | Reflective/motivational writing tone — informal but profound, blending personal gratitude with broader life philosophy about nature and human purpose.<br><br>The quote is being completed mid-sentence, echoing the earlier "magic, wonder and beauty" motif; it builds toward a concluding, aspirational statement about divine artistry or nature's purpose.<br><br>"artistry" is the final word, functioning as the object of "perceive," requiring a continuation — likely a relative clause or verb phrase completing the observation, e.g., "artistry in every moment" or "artistry that surrounds us." |
| control RL (λ=0) | 126 | 0.515 | Poetic/reflective prose rhythm: flowing, elevated language ("sigh," "tremble," "whispers," "breathtaking") guides continuation of lyrical tone.<br><br>Philosophical meditation on nature and existence: the text contrasts urban alienation with natural wonder, building toward a concluding transcendental statement about divine beauty.<br><br>Final token "artistry" ends an incomplete quote structure ("the artistry"), requiring a noun phrase or prepositional complement completing the metaphor — likely "in every moment" or "of life" or similar grand philosophical phrase. |
| λ=0.001 | 92 | 0.571 | Reflective, poetic prose about nature's beauty and presence, establishing philosophical tone throughout.<br><br>The quoted phrase "see the artistry" is mid-sentence, requiring completion with a noun or object — likely "in every leaf," "of nature," or similar.<br><br>Final token "artistry" ends an incomplete direct quotation ("see the artistry"), demanding immediate continuation of this noun phrase or clause. |
| λ=0.002 | 74 | 0.457 | Poetic/reflective prose style maintained throughout, blending nature with spiritual wonder.<br><br>Narrative momentum: the passage builds toward appreciating nature's artistry as evidence of something greater.<br><br>Final token "artistry" is mid-sentence, requiring a closing quotation or continuation describing what is being observed and felt. |
| λ=0.006 | 33 | 0.420 | poetic nature reflection tone<br><br>celebration of natural beauty<br><br>"artistry" requires completion |
| λ=0.015 | 24 | 0.237 | reflective tone<br>nature writing<br>"artistry" |
| λ=0.03 | 17 | -0.031 | appreciate the artistry |

### Sample 78
> …he next release! Tally-Bally ho I say, as I recharge my snifter and eagerly

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 133 | 0.714 | The author is building toward completing the sentence with an enthusiastic, anticipatory clause expressing eagerness — "eagerly" signals an immediate action phrase is expected, like "await the publication," "look forward to," or "wait for the release of."<br><br>The narrative follows a consistent enthusiastic, informal tone of excitement and anticipation throughout, so continuation will reflect this voice.<br><br>"eagerly" is the final word, an adverb modifying the upcoming main clause; it requires an active verb phrase describing what the narrator is eagerly doing (e.g., "awaiting," "watching," "waiting"). |
| control RL (λ=0) | 132 | 0.766 | Narrative momentum of anticipation/enthusiasm: the text has built up excitement about the books, concluding with "eagerly" demanding continuation of eagerness-themed actions.<br><br>First-person blog voice ("I"), informal yet enthusiastic tone ("can't wait," "dare not say") throughout, guiding continuation to match that register.<br><br>Final word "eagerly" is an adverb mid-clause ("and eagerly"), requiring an immediate verb phrase describing what the narrator is eagerly doing — most likely "awaiting," "looking forward to," or "anticipate" the book's arrival. |
| λ=0.001 | 88 | 0.708 | Enthusiastic book review momentum building toward a concluding anticipation statement. The narrative describes reader eagerness for the sequel.<br><br>List/description pattern established: books are being praised, details given, now transition to reader reaction.<br><br>Final token "eagerly" opens a participial phrase requiring completion, strongly predicting "await" or similar anticipation word followed by the next book/release. |
| λ=0.002 | 73 | 0.722 | Narrative momentum building toward the author's eagerness to read the upcoming series.<br><br>The article has followed a pitch structure: setup, characters, series premise, now wrapping up.<br><br>"eagerly" ends mid-clause, requiring completion with an infinitive verb like "await" or "prepare." |
| λ=0.006 | 24 | 0.203 | book review narrative<br>anticipation building<br>and eagerly |
| λ=0.015 | 25 | 0.515 | book review<br>excitement<br>"eagerly" |
| λ=0.03 | 13 | 0.461 | and eagerly |

### Sample 79
> …there limitations to the colors and patterns available for colored concrete

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 125 | 0.295 | FAQ list structure: a Q&A format is established, with each question followed by a brief explanatory answer, continuing the pattern of numbered/subscripted FAQs seen in the intro.<br><br>Domain/genre signals: construction/materials industry marketing copy explaining concrete products, maintaining informational but promotional tone throughout.<br><br>Final token is mid-sentence, completing "how much color concrete" — this is a partial question or phrase beginning an FAQ entry, immediately requiring a verb or continuation like "can be achieved" or "is needed," consistent with the numbered question format seen in the intro. |
| control RL (λ=0) | 138 | 0.603 | FAQ/Q&A format established throughout: alternating bold questions with detailed answers, predicting continued parallel structure with specific elaboration.<br><br>The text is addressing "cost factors" in concrete resurfacing — specific cost ranges, material types, and influencing variables are expected next to complete the "factors affecting cost" question.<br><br>Final token "concrete" ends an incomplete question mid-phrase: "What factors affect the cost of stamped concrete [resurfacing]" — the next tokens must close this question (likely "resurfacing?") and then provide a detailed cost-variability answer listing variables like material, labor, design complexity. |
| λ=0.001 | 83 | 0.401 | FAQ/Q&A format established, with repeated "How?" questions answered sequentially, so a concluding response is expected.<br><br>The answer to the final question about color variety must now follow, contrasting with the previous point about limited options.<br><br>Final token ends mid-phrase "concrete," requiring completion of the noun phrase before stating that color options are limited or restricted. |
| λ=0.002 | 70 | 0.528 | FAQ format with consistent question-answer structure signals next tokens continue answering.<br><br>The text is mid-answer listing pros/cons, continuing the pattern of balanced elaboration.<br><br>Final token ends mid-sentence after "concrete" — an incomplete question requires a period or continuation before the next FAQ entry. |
| λ=0.006 | 34 | 0.524 | FAQ Q&A format<br><br>completing the answer about cost<br><br>"concrete" ends the question |
| λ=0.015 | 29 | 0.347 | FAQ format<br>concrete cost<br>"what is the cost of stamped concrete" |
| λ=0.03 | 17 | 0.415 | what are the options for concrete |

### Sample 80
> …2:28 mark to push the lead to 4-0. Moments later, Scraper tallied her third

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 113 | 0.656 | Sports match recap genre with chronological game narrative, detailing possessions, scoring sequences, and player actions step by step.<br><br>Scoring chain momentum: the text is mid-sequence of consecutive scoring possessions, having described a second score before a third goal attempt, requiring completion of that action.<br><br>"third" is the final token — an ordinal adjective modifying an immediately expected noun, almost certainly "goal" (e.g., "third goal"), as the sequence tracks multiple consecutive possessions leading to more scoring opportunities for Yale. |
| control RL (λ=0) | 128 | 0.756 | Sports game recap format, tracking each player's scoring sequence chronologically throughout the match.<br><br>The narrative is building toward a cumulative goal tally for Maura O'Connor — the "third" implies her third goal is about to be described with specific match details (time, circumstance, type of assist, score progression).<br><br>"third" is a superlative ordinal mid-noun-phrase, requiring an immediate noun like "goal" or "score," followed by contextual details about the play (e.g., "goal off a..."), consistent with the journalistic game-report register used throughout. |
| λ=0.001 | 88 | 0.750 | Sports game recap narrative following chronological action sequence, detailing scoring moments by player.<br><br>List of scoring players and moments building: Biermann's first goal already mentioned, now Kelleher's third goal is in progress.<br><br>Final token "third" ends mid-sentence describing Kelleher's third goal, immediately requiring a noun like "goal" followed by scoring details or opponent. |
| λ=0.002 | 77 | 0.651 | Game recap narrative momentum: scoring plays are being detailed sequentially for each team.<br><br>Repetition/continuation pattern: each player's goal is described with setup, then score attribution.<br><br>Final token "third" is part of "third goal of the game," immediately requiring the word "goal" and likely a period/continue. |
| λ=0.006 | 31 | 0.425 | sports game recap narrative<br>scoring progression sequence<br>"third" requires a score detail |
| λ=0.015 | 22 | 0.414 | score recap<br>player highlights<br>"third" |
| λ=0.03 | 12 | -0.154 | third |

### Sample 81
> …secutors, the couple from 2004-2010 sold inventory from the Buttery Co. LLP

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 128 | 0.531 | Legal/regulatory news article following a structured reporting format (who, what, why, impact), requiring continuation of the business/ownership structure description.<br><br>Specific legal consequences are being enumerated: "criminal charges, civil complaints, administrative penalties" — the sentence introduces penalties and needs completion with specific cases/examples.<br><br>The final word "LLP" closes a legal entity name ("Beeplease Consulting Services LLP"), part of a sentence mid-construction that must specify the consequences or context of this partnership's failure, likely referencing the $1.3 million debt or a subsequent legal finding. |
| control RL (λ=0) | 111 | 0.524 | Local news article about a local business story, establishing context and narrative momentum toward the legal/political dispute.<br><br>The article has introduced both parties and is now detailing the plaintiffs' allegations, so specific grievances or legal claims are expected next.<br><br>"The company, B.B. Rogers LLP" ends mid-phrase with an organization name, requiring a verb phrase or prepositional clause completing what the company did (e.g., "sued," "was alleged to," "filed a complaint against"). |
| λ=0.001 | 111 | 0.604 | Legal/crime reporting genre with factual, chronological news article style throughout.<br><br>Narrative momentum: case details being established — defendant, charges, and context are being elaborated with "was accused of stealing more than $22,000 from B&B Accounting and Advisory LLP."<br><br>Final token "LLP" closes the company name mid-sentence, immediately requiring continuation of what happened to the LLP (e.g., "by embezzling" or a relative clause describing theft). |
| λ=0.002 | 67 | 0.587 | News article narrative building toward legal/financial consequences of the scheme.<br><br>Established pattern of presenting facts about the Ponzi scheme and its victims.<br><br>Final token "LLP" closes a corporate name mid-sentence, immediately requiring continuation describing the victims' connection to the company. |
| λ=0.006 | 26 | 0.461 | legal dispute narrative<br><br>company name continuation<br><br>" LLP" closing |
| λ=0.015 | 24 | 0.408 | legal case summary<br>business partnership<br>"LLP" |
| λ=0.03 | 160 | — | _(extraction failed)_ |

### Sample 82
> …should load it when run time (using source code) PS. i attached my original

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 119 | 0.683 | Technical bug report/bug-tracker format with structured sections (title, problem description, logs, user input) establishing formal reporting conventions throughout.<br><br>The user is mid-sentence describing what their original code looked like before making the change, requiring continuation of the original code example to demonstrate the specific line being modified.<br><br>"my original" is the start of a noun phrase mid-sentence, almost certainly continuing as "my original code" or "my original code follows" or "my original version," constrained by the bug-description context established. |
| control RL (λ=0) | 132 | 0.735 | User is sharing a code snippet (a simple JOptionPane confirmation dialog) and has now posted their "original" code — implying they will show the full original Java code for review/testing purposes.<br><br>The question context ("why it doesn't work") expects explanation of why the code isn't functioning, likely due to incorrect JOptionPane setup or missing logic (e.g., no action on "OK" button).<br><br>"my original" ends the sequence as the start of a noun phrase, immediately requiring a noun like "code" or "source" followed by a colon or description of what the original code looks like. |
| λ=0.001 | 80 | 0.753 | Developer seeking assistance with a specific Java Swing GUI problem, expecting code clarification or solution guidance.<br><br>Narrative shift: speaker is sharing their own code, implying a code snippet or implementation details follow.<br><br>Final token "original" begins a clause describing what they posted — next tokens must complete this phrase, likely "code" or "source code." |
| λ=0.002 | 62 | 0.781 | User providing Java code context and asking for help modifying behavior on button click events.<br><br>The narrative is mid-sentence explaining attachment/upload context.<br><br>"my original" ends a clause requesting code, strongly implying "code" or "Java file" follows. |
| λ=0.006 | 30 | 0.661 | code-sharing context<br>explaining modifications<br>"my original" refers to shared code |
| λ=0.015 | 23 | 0.644 | code explanation<br>original code<br>"my original" |
| λ=0.03 | 13 | 0.398 | my original |

### Sample 83
> …locating distribution located at all four corners of Glenmore Trail and 37 

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 143 | -0.110 | Structured list continuation: each bullet describes a specific route with numbered leg, direction, street names, and distances — this pattern repeats throughout.<br><br>The bullet point is mid-sentence, listing a numbered road segment ("leg 3") with a specific street name beginning "49th Ave," requiring completion with direction (e.g., "S"), then a crossing street, direction, and distance (e.g., "to 43rd Ave S").<br><br>Final token "49th Ave" is an incomplete proper noun mid-list-item, immediately requiring a directional modifier like "S" to complete the street name before proceeding to the destination street and distance. |
| control RL (λ=0) | 128 | -0.192 | Structured address/listing format: the document presents an official address with street, city, state, zip, and directional navigation instructions.<br><br>Repetition/continuation pattern: directional navigation text has been listing street intersections and routes, establishing a consistent "turn on X street at the corner of Y and Z street" pattern throughout.<br><br>Final token "39" is the beginning of a street intersection identifier (likely "39th Street"), mid-sentence within an ongoing directional instruction, immediately requiring a street name like "Street" or "Avenue" to complete the noun phrase. |
| λ=0.001 | 98 | -0.038 | Geographic event listing pattern: multiple road closures being enumerated sequentially with street names and dates.<br><br>Structured event details: location (San Diego, SD) establishes format of specific streets and dates for each closure.<br><br>Final token "62nd St & 40th St" is mid-sentence within a list of road closures; next tokens must complete the street name pair with "and" plus additional street details or closing information. |
| λ=0.002 | 74 | -0.187 | Geographic enumeration pattern listing specific road names in the area.<br><br>Address/address details being compiled for multiple locations.<br><br>Final token ends mid-phrase "route 32 and route 42 connecting to glacier" — next tokens complete the street name with "glacier [direction] street" or similar suffix. |
| λ=0.006 | 41 | -0.047 | Geographic event listing format<br><br>Road infrastructure details pattern<br><br>"64th and 66th street and" requires continuation of address |
| λ=0.015 | 160 | — | _(extraction failed)_ |
| λ=0.03 | 20 | 0.053 | Route 17 and 92 |

### Sample 84
> …ABOUT MY PROCEDURE! Dr. Alan J. Bauman is a surgically trained M.D. and one

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 122 | 0.505 | Testimonial/review genre with positive sentiment momentum: "I am extremely pleased" sets expectation for a glowing, enthusiastic endorsement about the provider.<br><br>Narrative arc of transformation: the review promises "the most dramatic change in my life," requiring elaboration on how the surgery achieved that dramatic result.<br><br>Final token "one" begins an incomplete noun phrase — likely "one of the best," "one of the most skilled," or "one who" — demanding a positive superlative or adjective completing the description of Dr. Burchfield's qualities. |
| control RL (λ=0) | 142 | 0.521 | Medical/doctor testimonial format signals a personal endorsement narrative, expecting positive outcome language. The testimonial structure follows patient praise for Dr. Tashjian's rhinoplasty results.<br><br>Repetition/continuation pattern: "Dr. Tashjian and one" introduces a paired contrast or completion — "one" likely continues as "one of the best" or "one of the most skilled," completing a superlative phrase praising the surgeon's expertise.<br><br>Final token "one" is the start of an incomplete superlative construction ("one of the..."), requiring a noun phrase immediately following, almost certainly "best" or similar adjective. |
| λ=0.001 | 86 | 0.456 | Testimonial review format established, requiring completion of personal endorsement of Dr. Laskin.<br><br>Personal narrative voice ("I am...I was") momentum building toward describing Dr. Laskin's qualifications and positive experience.<br><br>Final token "one" begins a superlative phrase ("one of the few/one who..."), immediately requiring a completing noun or adjective phrase. |
| λ=0.002 | 73 | 0.463 | Patient review/testimonial format establishes personal, grateful tone throughout.<br><br>Narrative momentum toward describing Dr. Baskin's qualifications and bedside manner.<br><br>Final token "one" begins a superlative phrase "one of the best," requiring continuation with a noun like "of the best surgeons in the area." |
| λ=0.006 | 37 | 0.496 | Testimonial review format<br><br>Dr. Fazio's credentials described<br><br>"and one" requires completion of a paired phrase |
| λ=0.015 | 27 | 0.322 | testimonial quote<br>Dr. Kao's description<br>"and one" |
| λ=0.03 | 15 | 0.103 | Dr. and one |

### Sample 85
> …ss in Israel, and especially on the ability to cover the security apparatus

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 126 | 0.633 | Investigative journalism article on Israeli military censorship, establishing a pattern of expert testimony with quotes and attributions to support claims.<br><br>Quoted speech from Professor Ben-Meir is mid-sentence, completing her argument about censorship's impact on Israeli society; the quote's closing phrase is grammatically incomplete, requiring continuation of her quoted remarks.<br><br>The final token "apparatus" ends an unclosed direct quotation mid-sentence; the next tokens must complete the thought, likely adding a closing punctuation mark or further elaboration from Ben-Meir before returning to attribution. |
| control RL (λ=0) | 127 | 0.651 | News article reporting a formal accusation (Knesset vote, government indictment), establishing journalistic tone with formal register throughout.<br><br>The article follows an inverted pyramid structure — key facts (veto, charges, Knesset) stated upfront; next expected content elaborates on the indictment's scope or background.<br><br>Final token "apparatus" ends mid-phrase "the security apparatus" — an incomplete noun phrase requiring continuation, likely "and its members" or a relative clause specifying which entity or official (e.g., "of the government" or "of Prime Minister Netanyahu"). |
| λ=0.001 | 100 | 0.679 | News article reporting on an Israeli Knesset hearing, establishing formal journalistic tone throughout.<br><br>Narrative momentum: the article has outlined the committee's concerns and is mid-sentence detailing the Knesset members' stated rationale for their request.<br><br>Final token "apparatus" ends an incomplete clause about transparency demands; next tokens must complete the phrase, likely with punctuation or continuation such as "in the security agencies" or similar elaboration. |
| λ=0.002 | 76 | 0.752 | Journalistic investigative article pattern, presenting criticism of Yuli Tamir's committee role.<br><br>The argument builds toward exposing limitations in parliamentary oversight of the security apparatus.<br><br>Final token "apparatus" is mid-sentence within a direct quote, requiring completion — likely adding "and its activities" or "in this country." |
| λ=0.006 | 36 | 0.640 | Journalistic report on Knesset investigation<br><br>explaining committee's mandate<br><br>"the security apparatus" requires completion |
| λ=0.015 | 27 | 0.558 | journalist report<br>parliamentary inquiry<br>"the security apparatus" |
| λ=0.03 | 14 | 0.478 | the security apparatus |

### Sample 86
> …ed to compact the repair material using a mechanical roller. The Department

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 107 | 0.736 | Official press release genre with formal institutional language, maintaining consistent quote-and-attribution structure throughout.<br><br>The Department of Transport spokesperson's statement is mid-sentence, directly responding to the accident inquiry about their policy or safety measures regarding bus driver fitness.<br><br>Final token "The Department" opens a new sentence continuing the spokesperson's quote, likely elaborating on the investigation, policy response, or assurance about safety standards (e.g., "The Department of Transport has undertaken/investigated..."). |
| control RL (λ=0) | 122 | 0.662 | Formal regulatory/administrative document style, with official body statements and disciplinary findings following a structured press release format throughout.<br><br>Narrative momentum: the quote is mid-sentence, presenting the Department's formal response to the disciplinary matter, continuing a pattern of official reassurances and procedural commitments.<br><br>Final token "Department" ends an incomplete clause — "The Department" strongly constrains the next tokens to a verb or noun phrase continuing the official statement, likely something like "has confirmed" or "has a firm zero-tolerance policy" regarding such incidents. |
| λ=0.001 | 94 | 0.754 | Official response format (FAQ structure) signals a formal institutional reply continuing to address concerns about the accident.<br><br>The quote is mid-sentence, establishing the Department of Transport's position on road safety standards; a commitment or assurance is expected to follow.<br><br>Final token "Department" opens a new clause within the direct quote, requiring immediate continuation like "of Transport" or "of Transport regrets/notes/apologises." |
| λ=0.002 | 68 | 0.622 | Formal press release/news bulletin style with bureaucratic language from a safety regulator.<br><br>Statement continuation expected: the Department's position or response on the incident is being elaborated.<br><br>Final token "The Department" begins a new clause continuing the formal regulatory response, immediately expecting a verb phrase. |
| λ=0.006 | 32 | 0.566 | Regulatory response format<br><br>apology/liability statement structure<br><br>"The Department" continues formal statement |
| λ=0.015 | 23 | 0.520 | official statement<br>regulatory response<br>"The Department" |
| λ=0.03 | 12 | 0.415 | The Department |

### Sample 87
> …Italian cook.” He said he hopes to see hearty entrees round out the bánh mì

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 148 | 0.570 | Restaurant review genre following structured format: executive summary then detailed critique, now closing with final verdict on dishes. The review has covered menu items in sequence.<br><br>The text has been systematically evaluating menu offerings, having addressed specials, main courses, sides, and entrees; dessert has been omitted, implying it's the next expected category in review structure.<br><br>Final token "bì" completes the term "bánh mì," a Vietnamese sandwich — the review is mid-sentence discussing how bánh mì is used as a vehicle, strongly constraining next tokens to continuation of this thought (e.g., "bánh mì at..." or elaborating on its role in the menu). |
| control RL (λ=0) | 131 | 0.440 | Ongoing restaurant review structure following a consistent pattern: describe dish, technique, texture, flavor — this pattern strongly predicts continuation of the bahn mi description.<br><br>The sentence ending "added to the bahn mi" promises elaboration on what is being added — ingredients or toppings — mirroring the salad's "added to the salad" structure already described.<br><br>Final token "bahn mi" ends an incomplete clause mid-thought; it is the object of "added to," requiring a prepositional complement or continuation phrase describing the addition (e.g., "at Pho La Lai"). |
| λ=0.001 | 96 | 0.288 | Restaurant review narrative continuing a list of menu innovations and culinary techniques introduced so far.<br><br>The review follows a pattern of describing new offerings, alternating between techniques and dishes, suggesting more examples follow.<br><br>Final token "bì" completes "banh mi," an incomplete phrase mid-clause listing menu additions; next tokens must complete the dish reference and likely continue with what the bì is paired with or how it was modified. |
| λ=0.002 | 86 | 0.657 | Article profiling Austin's restaurant scene, detailing new openings and trends.<br><br>Narrative momentum: description of Bao Brothers' menu items mid-list, with "bánh mì" construction still unresolved.<br><br>Final token "bánh mì" is an incomplete noun phrase ("to the bánh mì ___"), requiring a complement like "sandwiches" or similar dish. |
| λ=0.006 | 31 | 0.466 | Restaurant food descriptions<br>menu item elaboration<br>"bánh mì" needs completion |
| λ=0.015 | 25 | 0.508 | restaurant review<br>menu items<br>"bánh mì" |
| λ=0.03 | 16 | 0.400 | the bò mì |

### Sample 88
> …not yet realized that the dissertation (technically labeled as a thesis) is

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 139 | 0.516 | Argumentative thesis establishing a paradox: the work is "well known" yet "inconspicuous," requiring explanation of why this contradiction holds true despite recognition.<br><br>The text is building toward a pivotal contrast about the work's accessibility versus scholarly recognition, with the "but" pivot signaling a key claim about why it has flown under the radar despite its importance.<br><br>"the work is" is an incomplete predicate at the sentence's end, requiring a descriptive adjective phrase — almost certainly explaining the work's obscurity, e.g., "rarely taught," "inconspicuous," or similar continuation of the established paradox. |
| control RL (λ=0) | 114 | 0.641 | Academic introduction building toward a revelation about a minor work's significance, establishing contrast between obscurity and value.<br><br>The phrase "the treatise, though little-known and seldom-cited, is" sets up a positive evaluative clause asserting the text's importance or scholarly significance.<br><br>Final token "is" opens a predicate complement requiring an adjective or noun phrase — most likely "an important," "significant," "an overlooked," or similar evaluative descriptor affirming the treatise's value despite its obscurity. |
| λ=0.001 | 94 | 0.703 | Academic commentary establishing value of a lesser-known scholarly work ("The Origin of Ideas"), building toward its significance.<br><br>The phrase "that the treatise is" sets up an evaluative claim about the work's importance, likely "a valuable resource," "often overlooked," or "remarkably insightful."<br><br>Final token "is" opens a predicate requiring completion of the evaluative description of the treatise's significance. |
| λ=0.002 | 75 | 0.659 | Academic/forensic critique establishing the work's historical significance and importance.<br><br>Argumentative momentum: thesis building that the essay's significance outweighs its flawed status.<br><br>Final token "is" opens a predicate clause requiring a description of the essay's importance or value, e.g., "one of the most valuable..." |
| λ=0.006 | 30 | 0.563 | academic essay tone<br><br>argument about scholarly value<br><br>"the text is" requires description |
| λ=0.015 | 26 | -0.173 | academic reference<br>bibliographical note<br>"this essay is" |
| λ=0.03 | 14 | 0.339 | the thesis is |

### Sample 89
> …allation. You will find more detailed information and advices for an energy

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 122 | 0.678 | Listicle/guide format with enumerated benefits of energy-efficient school buildings, maintaining consistent imperative instructional tone throughout.<br><br>Sequential benefit enumeration pattern established: "Save money," "Enhance quality of learning," "Better health and comfort" — the list is clearly ongoing, with several more items likely remaining.<br><br>Final token "an energy" begins a noun phrase mid-sentence introducing another enumerated benefit or resource, strongly constraining next tokens to a compound adjective like "efficient" or "effective" followed by a noun such as "school" or "efficient building." |
| control RL (λ=0) | 127 | 0.687 | Bulleted list structure established with four dash-separated items; four items promised, third still in progress, requiring completion and a fourth.<br><br>The sentence "An energy" is mid-phrase, beginning the third list item; the word "energy" strongly signals "efficient" or "efficient home" as the expected continuation, consistent with the document's repeated theme of energy efficiency in housing.<br><br>The final token "energy" is the start of an incomplete noun phrase, almost certainly completing as "efficient [home/smart building/etc.]" to describe what the training resource covers. |
| λ=0.001 | 89 | 0.659 | Instructional guide structure: text has been giving recommendations, now concluding with a final section on energy efficiency measures.<br><br>List/continuation pattern: the document has consistently enumerated strategies (examples, steps, options), suggesting more specific measures follow.<br><br>Final token "an energy" begins a noun phrase requiring completion, most likely "an energy-efficient" followed by a building/operational feature. |
| λ=0.002 | 80 | 0.732 | List-based document pattern: structured tips with headers followed by detailed sub-sections.<br><br>Continuation of a "Guidelines for an energy" header, expecting a policy document completing a standard institutional title.<br><br>Final token "energy" is mid-phrase within a new header, immediately requiring a noun like "efficient building" or "efficient operation." |
| λ=0.006 | 31 | 0.673 | practical guide format continues<br><br>energy efficiency context established<br><br>"an energy" requires completion |
| λ=0.015 | 25 | 0.633 | energy efficiency topic<br>practical tools<br>"an energy" |
| λ=0.03 | 13 | 0.414 | an energy |

### Sample 90
> …l: [email protected] Akhil Vaidya – Footprint Global Communications Cell: 9

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 122 | 0.609 | Structured metadata table pattern: pipe-delimited key-value pairs like "Category\|Education," "Platform\|Android/iOS" establish repeating template rows.<br><br>List of contact information details: "Official Website," "Phone No" suggest more fields like Email or Contact Us link may follow in the same table format.<br><br>The final token "9" is the beginning of a numeric phone number value (likely "+91" or "+9"), requiring continuation as digits completing the mobile/landline number, likely followed by "\|\|" to close the table cell. |
| control RL (λ=0) | 106 | 0.706 | Contact information list format: a structured phone number is being given, requiring continuation of digits following "9".<br><br>Directory/contact page structure: each entry includes name, title, department, and phone, suggesting the next tokens complete this final entry's phone number.<br><br>The final token "9" is the beginning of an Indian mobile phone number (starting with 9), so the immediate next tokens must be 5-7 additional digits completing a valid 10-digit number. |
| λ=0.001 | 87 | 0.703 | Contact information pattern: listing phone numbers and social media handles following a "contact us" section.<br><br>Sequential number continuation: the sequence "9" is mid-number, requiring remaining digits to complete the Indian mobile number (10 digits total).<br><br>Final token "9" is the start of an incomplete phone number, immediately constraining next tokens to be more digits completing the number. |
| λ=0.002 | 67 | 0.640 | Contact details format: phone number being listed after "9"<br><br>Institutional contact page pattern with structured fields (address, website, phone)<br><br>The final token "9" is the start of a 10-digit Indian phone number, immediately requiring continuation with more digits |
| λ=0.006 | 28 | 0.171 | Contact info pattern<br>phone number sequence<br>"9" starts next digit |
| λ=0.015 | 22 | 0.196 | contact details<br>phone number<br>"9" |
| λ=0.03 | 12 | -0.694 | 9 |

### Sample 91
> …ing life skills along with healthy habits is more important by far than any

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 131 | 0.785 | Motivational/inspirational register established throughout, using sports metaphors to build toward a concluding life-lesson about athletics and character-building.<br><br>The essay follows a pattern of "what [something is]—so why it matters," building from specific athletic examples toward universal values; a culminating conclusion about intrinsic character over extrinsic success is expected next.<br><br>Final token "any" opens a superlative construction ("any ___") completing the contrast "more important than any ___," requiring a noun phrase like "trophy," "achievement," or "external success" to finish the rhetorical statement. |
| control RL (λ=0) | 133 | 0.758 | Argumentative momentum: a motivational/advocacy essay building toward a conclusion about character, teamwork, and sportsmanship outweighing competition.<br><br>Parallel structure contrast: "more than any" sets up a superlative comparison, contrasting the intangible qualities of character/teamwork against the tangible rewards of athletic achievement (e.g., trophies, victories).<br><br>Final token "any" is the start of a comparative clause ("more than any [X]"), requiring a noun phrase completing the comparison — most likely a specific competitive or material goal such as "win," "trophies," or "individual accolades." |
| λ=0.001 | 89 | 0.735 | Argumentative essay building toward a moral conclusion about youth sports and character development.<br><br>Contrastive structure: "any" at the end sets up completing a "not as valuable as..." comparison, requiring a noun phrase like "win" or "medal."<br><br>Final token "any" is mid-phrase in a contrasting superlative construction, requiring immediate continuation of the comparison object. |
| λ=0.002 | 80 | 0.682 | Argumentative essay building toward a concluding moral about sportsmanship and character over competition.<br><br>The "no comparison" rhetorical setup establishes a contrast requiring completion between sportsmanship and athletic achievement.<br><br>Final phrase "any" ends an incomplete clause comparing sportsmanship to competition outcomes — next tokens must finish the contrast, likely "other prize" or "result." |
| λ=0.006 | 32 | 0.377 | sports values argument<br><br>contrast between sports and competition<br><br>"any" completes "none of any" |
| λ=0.015 | 22 | 0.384 | sports quote<br>life lessons<br>"any" |
| λ=0.03 | 12 | 0.172 | any |

### Sample 92
> …ilent auction. Donated works not sold at the fundraiser can be retrieved by

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 120 | 0.434 | Bureaucratic/organizational announcement style with bullet-pointed rules and procedural guidelines, established throughout — formal institutional language expected to continue.<br><br>The document systematically lists submission rules in sequence (title requirements, size requirements, entry fee, shipping details), with each rule building toward deadline information and pick-up instructions.<br><br>Final token "by" closes a phrase beginning "pick-up by" — requires a temporal completion, almost certainly a deadline date, time, or condition such as "April 15th" or "the exhibitor." |
| control RL (λ=0) | 123 | 0.659 | Institutional bulletin format with bullet-point sections covering admission rules — the pattern suggests a closing procedural note is being completed.<br><br>The phrase "will be retrievable by" sets up a specific group of people or roles (e.g., "original owners," "applicants," "designated persons") who can access submissions once the deadline passes.<br><br>Final token "by" is a preposition completing a passive construction, requiring an immediate noun phrase or pronoun identifying who can retrieve submissions — e.g., "by original owners" or "by the applicant." |
| λ=0.001 | 84 | 0.634 | Organizational announcement format establishes procedural detail pattern, guiding toward retrieval/contact instructions.<br><br>The sentence structure "will be retrieved by" requires a subject (e.g., "artist" or "the artist") completing the retrieval process.<br><br>Final token "by" is a preposition mid-clause, immediately requiring a noun phrase specifying who retrieves the photos or contact method. |
| λ=0.002 | 69 | 0.404 | Bulleted list of drop-off/pick-up rules establishing formal institutional guidance throughout.<br><br>The list promises completion of a conditional rule about retrieving items.<br><br>Final token "by" opens a temporal/location clause, immediately requiring a time or method phrase like "the end of the school year." |
| λ=0.006 | 31 | 0.639 | event logistics description<br><br>pickup instructions continuation<br><br>"could be retrieved by" requires a subject |
| λ=0.015 | 25 | 0.514 | checkout procedures<br>artist instructions<br>"can be retrieved by" |
| λ=0.03 | 15 | 0.160 | will be returned by |

### Sample 93
> …by mistake and must hide her morally imperfect behavior and try to become a

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 132 | 0.501 | Narrative setup of an ironic premise: "brought into a world where good people are bad" establishes a dark comedic premise requiring a humorous resolution or twist.<br><br>The sentence structure promises completion of an expected phrase: "better person," "more virtuous individual," or similar — the irony demands a punchline revealing what the characters mistakenly believe about morality.<br><br>Final token "a" is an indefinite article beginning a noun phrase, requiring an adjective or noun immediately after — almost certainly a satirical, ironic word like "better person" or "virtuous being" to complete the setup's punchline. |
| control RL (λ=0) | 115 | 0.600 | Narrative momentum: the setup describes a sitcom premise, building toward explaining the protagonist's transformation arc.<br><br>Semantic expectation: the phrase "she must become a" strongly signals a role/character category, almost certainly "a better person" or "a morally upright/humane person" — the comedic twist premise established that she is a horrible person.<br><br>Final token: "a" is an indefinite article beginning a noun phrase completing "she must become a __", requiring an adjective or noun describing her moral character transformation. |
| λ=0.001 | 101 | 0.587 | Plot summary establishes a moral transformation arc, requiring completion of the character's redemption.<br><br>The phrase "turns out she needs to be a" sets up the ironic goal the protagonist must achieve — likely "better person" or "better human" consistent with the "good person" premise.<br><br>Final token "a" is an indefinite article requiring a noun phrase completing the character's necessary transformation, almost certainly "better person" or similar moral descriptor. |
| λ=0.002 | 62 | 0.506 | Narrative setup of irony/reversal: protagonist's good deed backfires on her<br><br>Sequential cause-effect structure building toward the comedic twist<br><br>"become a" is an incomplete predicate requiring a noun phrase describing the opposite of "bad person" |
| λ=0.006 | 34 | 0.507 | Narrative summary pattern<br><br>moral/character arc development<br><br>"become a" requires noun phrase |
| λ=0.015 | 27 | 0.533 | movie plot summary<br>moral transformation<br>"needs to become a" |
| λ=0.03 | 14 | 0.355 | become a |

### Sample 94
> …unit. Eliminating various chemicals needed in the manufacture of TFTs (such

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 122 | -0.059 | Ongoing metaphor/analogy pattern: "bulb" has been introduced as a substitute for "light" to emphasize solar lanterns, creating a parallel structure requiring completion.<br><br>The article consistently uses accessible analogies (e.g., "as simple as lighting a bulb") to explain technical concepts, making the solar analogy explanation the expected continuation.<br><br>Final token "light" is the start of the compound "light bulb," mid-word, requiring immediate completion with "bulb" before continuing with the parallel sentence structure established earlier ("as simple as..."). |
| control RL (λ=0) | 133 | -0.093 | Blog post format with structured metadata, author name, and topic tag preceding a teaser — narrative momentum points to completing a metaphorical or ironic expression.<br><br>The phrase "as bright as the light" is a well-known idiomatic setup requiring completion, almost certainly "as bright as the light at the end of the tunnel" — a cliché about hope or the future.<br><br>Final token "the light" is mid-phrase within the idiom, part of the established metaphor "as bright as the light [at the end of the tunnel]," constraining the next tokens to complete that near-universal expression. |
| λ=0.001 | 85 | -0.169 | Ongoing metaphor about LED lighting being "too bright/light," requiring completion of the idiomatic expression.<br><br>The blog post's critical but constructive tone about LED efficiency demands a balanced conclusion acknowledging improvements but still identifying weaknesses.<br><br>Final token "as" begins a comparative phrase ("as bright as"), constraining next tokens to complete the simile about LED brightness potentially being excessive. |
| λ=0.002 | 78 | -0.046 | Rhetorical question building toward critique of street lighting efficiency<br><br>Continuation of metaphor/analogy comparing traditional bulbs to LED technology<br><br>Final token "light" is part of the incomplete phrase "as bright as the light," strongly constraining next word to "as bright as the light bulb/lighting" to complete the comparison |
| λ=0.006 | 35 | 0.040 | analogy explanation in progress<br>metaphor completion expected<br>"as bright as the light" requires continuation |
| λ=0.015 | 23 | 0.129 | blog post<br>lamp metaphor<br>"the light" |
| λ=0.03 | 13 | 0.087 | the light |

### Sample 95
> … " An error âExtra variables on rhs: "denoFunDef"â occurs. I donât know why

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 119 | 0.517 | Question is being resolved by the user seeking help debugging a MATLAB/Optimisation problem — next tokens will explain the issue or ask for clarification.<br><br>The narrative "I think the code should be right, but don't know why" sets up a problem statement expecting elaboration, error messages, or a specific "why?" question.<br><br>The final token "why" closes a clause mid-thought, strongly constraining what follows: a question mark, punctuation ending this sentence, or a conjunction beginning an explanatory clause detailing the suspected issue. |
| control RL (λ=0) | 111 | 0.283 | Programming Q&A format (Stack Overflow-style): question raised, explanation given, "So why" signals an unresolved problem requiring explanation of code's issue.<br><br>Mathematical physics context: Lagrange multipliers used to enforce angular momentum constraint in a central potential problem, establishing domain knowledge expectations.<br><br>Final token "why" closes the "So why" transition clause, immediately requiring the answer/explanation of the bug — likely "is this not working?" or a question like "does my code fail?" |
| λ=0.001 | 97 | 0.611 | Code block just closed; author is now explaining confusion about code behavior and seeks clarification.<br><br>The phrase "I do not know why" signals a self-posed question requiring elaboration of the specific issue or problem observed.<br><br>Final token "why" closes an incomplete explanatory clause — next tokens must complete the question, likely explaining what unexpected behavior is occurring (e.g., "the output differs" or "the matrix behaves differently"). |
| λ=0.002 | 77 | 0.568 | Question about unexpected MATLAB code output, expecting a diagnostic explanation of why `0` appears instead of expected result.<br><br>Domain signals: MATLAB coding error, mathematical logic, matrix manipulation context.<br><br>Final token: "why" closes a question about a specific unexpected output, immediately expecting a closing question mark `?` then explanatory reasoning. |
| λ=0.006 | 27 | 0.612 | code debugging question<br>error explanation requested<br>"why" ends question |
| λ=0.015 | 22 | 0.242 | code error<br>syntax issue<br>"Why is" |
| λ=0.03 | 15 | 0.438 | don't know why |

### Sample 96
> …ncorporate the LSA technology as part of its personalization and in-surface

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 135 | 0.640 | Technical product description pattern: formal press release introducing Tesa's E-Code marking solution, requiring specific technical capabilities and applications.<br><br>List/continuation momentum: "full-color marking and in-surface" strongly implies a compound noun phrase completing the paired technology description, likely "in-surface engraving" or "in-surface decoration" — the standard industry term for this capability.<br><br>Final token "in-surface" is mid-phrase, part of an enumerated capability list after "full-color marking," requiring an immediate noun or hyphenated compound noun to complete the technical description of Tesa's product feature. |
| control RL (λ=0) | 125 | 0.693 | Press release/promotional format establishing a formal business announcement tone, with company name, product line, and executive quote already given.<br><br>Narrative momentum: the "Our mission is to help clients..." sentence is building toward describing capabilities; the list of services/technologies being enumerated requires completion.<br><br>Final token "in-surface" is part of the compound noun "in-surface decoration" (a standard industry term), directly requiring the completion of this specific technical phrase — almost certainly "decoration" or "display technology" — as the noun phrase concludes the sentence. |
| λ=0.001 | 89 | 0.596 | Business/marketing article establishing market opportunity for UV printing in consumer goods and retail.<br><br>List continuation pattern: "full-color printing, in-surface" signals a paired series of printing capabilities still being enumerated.<br><br>Final token "in-surface" is mid-phrase in a list, requiring immediate continuation with the noun it modifies, most likely "display" or similar UV printing application term. |
| λ=0.002 | 71 | 0.573 | Ongoing list of technologies being described — "in-surface" signals continuation of related terms.<br><br>Promotional/marketing tone about product capabilities and business advantages.<br><br>The final token "in-surface" is mid-phrase, part of a compound technology description requiring a noun or further modifier next. |
| λ=0.006 | 28 | 0.523 | Marketing/product description<br><br>technical specification context<br><br>"in-surface" requires completion |
| λ=0.015 | 24 | 0.540 | technical product description<br>display manufacturing<br>"in-surface" |
| λ=0.03 | 14 | 0.313 | in-surface |

### Sample 97
> …ght 2 Aug Ventura County Mountains Clear. Lows from the mid 60s to lower 70

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 113 | 0.316 | Weather forecast format established — structured daily summary pattern with temperature ranges and conditions expected.<br><br>The sequence has been listing forecast days sequentially (Wednesday → Thursday → Friday), so continuation of Saturday/Friday's temperature range and wind details is expected.<br><br>The final token "seventies" begins Friday's upper temperature range (35-50°F), requiring a closing "s" to complete "seventies," then wind conditions matching Wednesday/Thursday's 10-15 mph NW wind pattern. |
| control RL (λ=0) | 140 | 0.522 | Weather forecast format established, following standard meteorological structure (time periods, temperatures, precipitation, winds).<br><br>Temperature range completion: "60s and low 70" is mid-sentence, part of a dual temperature zone pattern seen earlier ("high 70s and low 80s"), requiring the number to be completed with "s" and likely a closing phrase like "in sheltered areas."<br><br>Final token "70" is the start of a truncated ordinal temperature descriptor ("low 70s"), directly requiring the letter 's' to complete the temperature expression, likely followed by a period or geographic qualifier. |
| λ=0.001 | 91 | 0.467 | Meteorological forecast format with structured temperature/condition details throughout.<br><br>Continuation of temperature range pattern: low 60s mentioned, high 70s/80s implied to complete the range.<br><br>Final token "70" is mid-number in a temperature range ("mid to upper 70s"), immediately requiring completion with "s" followed by closing parenthesis and forecast period. |
| λ=0.002 | 83 | 0.494 | Weather forecast format with structured temperature and condition ranges throughout.<br><br>Numerical temperature range being completed: "70s" signals the high bound of the range is coming.<br><br>The final token "70" is part of the temperature range "low 60s and high 70[s]," requiring a closing digit and unit before continuing the forecast. |
| λ=0.006 | 28 | 0.280 | weather forecast pattern<br><br>temperature range completion<br><br>"70" requires unit |
| λ=0.015 | 27 | 0.300 | weather forecast<br>temperature range<br>"60 to 70" |
| λ=0.03 | 15 | -0.200 | the 70 |

### Sample 98
> … reputation for paying out winnings in a timely manner. Finally, you should

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 125 | 0.751 | Listicle/guide structure with numbered tips: each section follows a heading with explanatory paragraph, establishing expectation of a parallel closing thought.<br><br>Semantic momentum: the final sentence begins "Before you sign up for any of these sites, you should" — this is a call-to-action imperative construction, implying actionable advice like reviewing terms, checking bonuses, or verifying legality follows.<br><br>Final token "should" opens a verb clause requiring a infinitive verb phrase (e.g., "carefully review," "ensure"), continuing the guide's practical, informative tone with reader-directed recommendations. |
| control RL (λ=0) | 127 | 0.721 | Numbered list structure (1-10) with consistent pattern: item number, bold topic, explanatory paragraph. Item 10 is the final entry being completed.<br><br>Each entry follows a formula: bold heading + ~3-4 sentences explaining the tip, implications, and caveats — item 10's paragraph is mid-completion.<br><br>"you should" opens the final imperative clause of item 10's explanation, immediately requiring a verb phrase (e.g., "research," "look for," "check") describing what players should do to assess credibility. |
| λ=0.001 | 99 | 0.799 | Instructional/advisory listicle structure: each sentence offers a tip for casino deposit methods, maintaining consistent advice format.<br><br>Final clause "You should" signals an imperative concluding recommendation, likely about verifying legitimacy, comparing sites, or final evaluation steps.<br><br>Final token "should" is the start of a new tip clause, immediately requiring a verb phrase continuing the checklist of consumer advice (e.g., "compare sites," "read reviews"). |
| λ=0.002 | 77 | 0.756 | Structured guide pattern: each section follows "should + verb + explanation" format consistently.<br><br>Enumeration of practical tips nearing completion — advice about bonuses, promotions, and payments has been covered.<br><br>"you should" opens a final imperative clause, requiring a verb phrase completing the recommendation (e.g., "read the terms and conditions"). |
| λ=0.006 | 31 | 0.666 | Guide format with numbered tips<br>Final recommendation expected<br>"you should" opens imperative clause |
| λ=0.015 | 24 | 0.724 | online casino guide<br>recommendations<br>"you should" |
| λ=0.03 | 13 | 0.435 | you should |

### Sample 99
> …an expected 1.7 million bbl draw. The market remains generally oversupplied

| model | tok | FVE | explanation |
|---|--:|--:|---|
| base (no RL) | 120 | 0.728 | Financial journalism covering oil market analysis, maintaining consistent pattern of presenting expert quotes and analyst views before elaborating on specific price predictions.<br><br>Narrative momentum builds toward explaining the supply/demand outlook after listing analyst forecasts; the bullish sentiment and production expectations suggest a discussion of market constraints or inventory data is forthcoming.<br><br>"The market is otherwise over supplied" ends mid-thought, completing a contrast structure ("otherwise" signaling a pivot or exception), immediately requiring elaboration on what "otherwise" refers to — likely inventory levels, global surplus, or demand concerns. |
| control RL (λ=0) | 135 | 0.755 | Financial news report structure: each paragraph introduces a new data point, market insight, or expert statement about crude oil supply/demand conditions.<br><br>Narrative momentum toward supply vs. demand analysis: the article consistently weighs supply growth (800k-1.4m bpd) against demand growth, building toward an evaluation of market balance or price outlook.<br><br>Final token "over supplied" ends an incomplete descriptive clause — the phrase "still oversupplied" strongly implies continuation with a prepositional phrase or subordinate clause, e.g., "and the market has room to absorb..." or "but demand suggests..." |
| λ=0.001 | 82 | 0.752 | Market analysis report tracking crude oil prices, supply/demand metrics, and trade dynamics throughout.<br><br>Narrative momentum: after listing supply conditions, the text pivots to a broader market outlook statement requiring elaboration.<br><br>Final phrase "is otherwise over-supplied" is mid-sentence, requiring completion describing continued over-supply conditions or their implications for prices. |
| λ=0.002 | 70 | 0.730 | Ongoing analysis of oil market fundamentals, expecting continuation of supply/demand outlook.<br><br>Narrative momentum: report building toward market outlook implications after detailing supply factors.<br><br>Final token "over supplied" ends a concluding clause mid-thought, immediately expecting elaboration on the market's oversupply condition. |
| λ=0.006 | 32 | 0.652 | Market analysis momentum<br>supply-demand balance discussion<br>"overall over supplied" requires completion describing conditions |
| λ=0.015 | 25 | 0.624 | market analysis<br>supply/demand dynamics<br>"over supplied" |
| λ=0.03 | 14 | 0.096 | still over supplied |
