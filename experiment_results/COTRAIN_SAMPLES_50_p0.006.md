# Co-train vs frozen — 50 full held-out samples (λ=0.006)

Matched by `idx` on the **same** held-out documents (rows 25k–26k, doc-disjoint from RL training). Both explanations are single temperature-1.0 samples. FVE shown is on the **frozen base-AR ruler** (apples-to-apples per sample); the co-trained *system* number (own-AR ruler) is higher in aggregate — see [`COTRAIN_COMPARISON.md`](COTRAIN_COMPARISON.md).

- **frozen** = AV trained with a frozen critic (original sweep), FVE 0.482 / 31.9 tok.
- **co-train** = AV + AR co-trained (`--train-critic`), FVE 0.451 base-AR / 0.507 own-AR / 26.0 tok.

Same pattern as λ=0.015 but milder (explanations are longer here): co-train drops the **genre/document-type framing** and the verbose "establishing…/introduces…" connectives (structural-framing words fall 61%→19% of samples), keeping the bare topic.

---

### idx 0

**Source** (last 300 chars): …, Stacy Clark. Clark is a continuum navigator in Tabitha’s central and western regions, which includes the cities of Grand Island and York, Nebraska. Clark joined Tabitha over 15 years ago, serving as an expert and guiding clients and families through the overwhelming process of Senior Care services

**Frozen** — 32 tok, FVE 0.613:
```
Narrative about program impact

explaining services provided

"services" closes sentence requiring continuation
```
**Co-train** — 26 tok, FVE 0.118:
```
introduction of role

organization description

"for families seeking..."
```

### idx 20

**Source** (last 300 chars): … to February 28th. Second, in March, VIP members will receive a free miniature Star Wars Republic Frigate with purchases of $50 or more. Third, free shipping is available for S@H orders from March 1st to March 15th. And finally, VIP members will receive double VIP points between March 25th and April

**Frozen** — 32 tok, FVE -0.021:
```
loyalty program benefits explained
yearly points threshold pattern
"to" begins completion phrase
```
**Co-train** — 28 tok, FVE -0.083:
```
loyalty program context

quote continuation

"becomes a member of"
```

### idx 40

**Source** (last 300 chars): … important for understanding future changes to ocean productivity brought about by a warming climate and changing environmental conditions.| |Rights:||All UHM dissertations and theses are protected by copyright. They may be viewed from this source for any purpose, but reproduction or distribution in

**Frozen** — 26 tok, FVE 0.016:
```
Academic repository metadata

Reproduction policy statement

distribution in
```
**Co-train** — 24 tok, FVE 0.101:
```
legal notice

reproduction policy

"distribution in"
```

### idx 60

**Source** (last 300 chars): …se Geometric Doormat This product is currently sold out. Strong fibers made from the husks of coconuts set the tone for a sustainable home. This durable doormat features a bold geometric pattern in a versatile muted turquoise color making it the perfect addition to any front door. - 18 x 30 in - 100

**Frozen** — 29 tok, FVE 0.737:
```
Product description format
material composition expected
"100" begins percentage
```
**Co-train** — 24 tok, FVE 0.484:
```
product description

material specification

"100"
```

### idx 80

**Source** (last 300 chars): …much better overall.” The Hawkeyes struck first at the 9:05 mark as Scraper scored to give Iowa a 1-0 lead. In the 17th minute, Scraper scored off a rebound to push the Hawkeye lead to 2-0. Sarah Drake tallied a goal at the 42:28 mark to push the lead to 4-0. Moments later, Scraper tallied her third

**Frozen** — 31 tok, FVE 0.425:
```
sports game recap narrative
scoring progression sequence
"third" requires a score detail
```
**Co-train** — 24 tok, FVE 0.387:
```
game recap narrative

scoring progression

"third"
```

### idx 100

**Source** (last 300 chars): …mmon to find clothing fiber, dirt and construction materials in computer facilities. Less common dust sources also exist. Natural events, such as volcanic eruptions and dust storms are some of the largest dust producing events possible. These events introduce a variety of different constituents into

**Frozen** — 38 tok, FVE 0.650:
```
Technical description of particulate matter

Continuation of cause-effect chain

"introduced into" requires a location/medium
```
**Co-train** — 24 tok, FVE 0.511:
```
technical exposition

dust particle effects

"into"
```

### idx 120

**Source** (last 300 chars): …way we perceive, the way we sense this life. The moonlight is playing a very important role. During the day we have sunlight. The sunlight brings information into our body. During the night, during the dark period, the information is digested. But undigested information is stored in our body. During

**Frozen** — 34 tok, FVE 0.757:
```
Ayurvedic moon cycle explanation

description of lunar activity phases

"During" opens a temporal clause
```
**Co-train** — 26 tok, FVE 0.741:
```
Ayurveda context

lunar cycle explanation

"During"
```

### idx 140

**Source** (last 300 chars): … no slot games rival fafafa™ gold progressives and levels. Will enjoy specific slots. Even better, you can buy specific totally free local casino dollars. If you are after real casino slots online that are incredibly quick and easy, the fafafa slot may well be the one for you. It's available at many

**Frozen** — 35 tok, FVE 0.641:
```
casino game promotional content

list of features and availability

"many" requires completion of a noun phrase
```
**Co-train** — 25 tok, FVE 0.793:
```
online casino content

game description

"available at many"
```

### idx 160

**Source** (last 300 chars): …purchase of clothing as well. If you are using coupon codes for clothing to shop online, look for free shipping offers to save even more. Free Express US Shipping on $100+ Enter code at checkout to save! BOGO Coupon - Buy One Get One 50% Off Everything Entrer code at checkout for savings. Save Up to

**Frozen** — 33 tok, FVE 0.715:
```
Promotional offer context
coupon/deal format continuation
"Save Up to" requires an amount
```
**Co-train** — 25 tok, FVE 0.662:
```
coupon codes listed

discount promotions

"Save Up to"
```

### idx 180

**Source** (last 300 chars): …. Walking performance was measured using the timed 25-ft walk (T25FW) and six-minute walk (6MW). Gait parameters were captured using a GaitRite™ electronic walkway. RESULTS: Aerobic capacity, balance, and knee-extensor asymmetry were associated with walking performance and gait in persons with MS (r

**Frozen** — 30 tok, FVE 0.565:
```
statistical reporting pattern

correlation coefficients expected

"(r" opens coefficient notation
```
**Co-train** — 23 tok, FVE 0.144:
```
statistical reporting

correlation results

(r
```

### idx 200

**Source** (last 300 chars): …lian summers. After successful international expansions in Dubai, Doha, Cannes, Istanbul and, most recently, Egypt’s North Coast, Scalini has opened in Bahrain, in association with Sevens Holding, the fastest growing luxury hospitality company in the GCC region. The restaurant is located at the five

**Frozen** — 33 tok, FVE 0.579:
```
hotel description in progress
detailed luxury property description
"the five" begins a noun phrase
```
**Co-train** — 25 tok, FVE 0.664:
```
hotel announcement

formal hospitality tone

"the five"
```

### idx 220

**Source** (last 300 chars): …sure, however, can come from all types of radioactive materials if they are inside the body. Once inside, much of the radiation energy will get absorbed in. Radiation Safety Training This course covers two main areas, the awareness of the safety problem and the legal framework. Firstly the awareness

**Frozen** — 35 tok, FVE 0.326:
```
Multiple unrelated content fragments

FAQ/casino content mixed together

"the awareness" begins a new section
```
**Co-train** — 23 tok, FVE 0.348:
```
training material

course content

"awareness"
```

### idx 240

**Source** (last 300 chars): … would be convincing shippers to spend the extra time coming upriver, and then, also convincing freight rail shippers to add the port as a regular scheduled stop so cargo could move quickly to markets west or north. "It is a bit of a chicken and egg problem," Hendrick said. Bnearing@nycap.rr.com • 5

**Frozen** — 29 tok, FVE 0.726:
```
news article format
phone number continuation
"5" is part of number
```
**Co-train** — 22 tok, FVE 0.615:
```
contact information

phone number

"5"
```

### idx 260

**Source** (last 300 chars): …us of DFS uncertain, and highlights separate legal issues in the potentially even more lucrative mobile sports betting market in New York. DFS may no longer be legal, but more traditional in-person sports betting on the outcome of actual sporting events is currently legal and operational at New York

**Frozen** — 30 tok, FVE 0.417:
```
sports betting coverage pattern

location/state context expected

"New York" requires completion
```
**Co-train** — 25 tok, FVE 0.394:
```
casino news reporting

location context

"New York"
```

### idx 280

**Source** (last 300 chars): …annel, cast iron, round bar, square bar, water pipe, gas pipe, plumbing pipe, STK steel pipe, stainless - Abrasive Grain - Grain Size * Products that meet your criteria are shown in Customers Who Viewed This Item Also Viewed Sumitomo Electric Industries | 5 star | 4 star | 3 star | 2 star | 1 star 3

**Frozen** — 29 tok, FVE 0.694:
```
Product review content
Customer ratings pattern
"3" starts a rating count
```
**Co-train** — 22 tok, FVE 0.391:
```
product review

rating summary

"3"
```

### idx 300

**Source** (last 300 chars): …ttery for a runtime of 60 minutes. The X2L LED Series easily mounts to most subcompact pistol rails utilizing Insight's patented Slide-Lock interface. Replacement LED kits will be available later this year. For more information, click here Request information about this product Other Recent Products

**Frozen** — 30 tok, FVE 0.311:
```
Product catalog listing
"Recently Added Products" section header
more product listings expected
```
**Co-train** — 26 tok, FVE 0.407:
```
product announcement

technical specs listed

"Other Recently Added Products"
```

### idx 320

**Source** (last 300 chars): …I am sitting here feeling very sad over the LODD of Ray Barrett who was killed when the truck he was responding to a motor vehicle crash in rolled off the road yesterday just north of Hicks Bay at Waihau Bay. Of even more concern for

**Frozen** — 31 tok, FVE 0.491:
```
Personal reflection tone
factual context established
"particular concern for" requires completion
```
**Co-train** — 27 tok, FVE 0.507:
```
personal reflection

concerns about firefighters

"Special concern for"
```

### idx 340

**Source** (last 300 chars): …s that we saw last year but actually below the long-term normal," he said. "The water into the landscape, the supply has really fallen off here over the last couple months." Rising lake levels have posed problems for some Michigan marinas and have threatened the loss of lakefront property during the

**Frozen** — 33 tok, FVE 0.706:
```
flood damage context established
recreational use implications
"during the" requires time phrase
```
**Co-train** — 26 tok, FVE 0.653:
```
flood damage context

property owners impact

"during the"
```

### idx 360

**Source** (last 300 chars): … to: - Increase your revenue and access new markets due to universal compliance and certification results (e.g. European and International Cybersecurity Standards, GDPR, Eurosmart IoT Cybersecurity Certification, CC, CSPN, Eurosmart Iot SCS, ISA/IEC 62443, ARM DSA, SESIP, IoT SF, etc.) - Reduce your

**Frozen** — 34 tok, FVE 0.633:
```
Bulleted benefits list continuing
security/quality risk reduction theme
"reduce your" ends phrase
```
**Co-train** — 25 tok, FVE 0.583:
```
benefits list

consultancy services

"reduce your"
```

### idx 380

**Source** (last 300 chars): …China. “The question of the trade imbalance is one that I think will be placed firmly on the table by the African delegates,” Sim Tshabalala, chief executive of South Africa’s Standard Bank Group, told reporters on Saturday. Critics also say Africa is loading itself up on Chinese debt that countries

**Frozen** — 30 tok, FVE 0.253:
```
Debt crisis narrative building
specific financial examples expected
"countries" requires completion
```
**Co-train** — 28 tok, FVE 0.333:
```
debt crisis context

Chad's sovereign debt burden

"countries"
```

### idx 400

**Source** (last 300 chars): …d||21/9/19 → 21/9/22| Bibliographical noteFunding Information: This work was supported by the National Research Foundation of Korea (NRF) funded by the Korea Government (Ministry of Science and ICT, MSIT) under Grant NRF-2020R1A2C3011697, and the Yonsei University Research Fund of 2021 (2021-22-0001

**Frozen** — 35 tok, FVE -0.151:
```
Bibliographic citation format
Research grant acknowledgment
Final token "20200401"
```
**Co-train** — 22 tok, FVE -0.532:
```
academic paper

research acknowledgment

final token
```

### idx 420

**Source** (last 300 chars): …for sustainable growth. Support from unions and public-education advocates won Jim Kenney the primary election, despite $7 million in outside spending for his opponent. Activist Linda Sarsour on the Arab American struggle against police brutality. What a serious desegregation policy might look like.

**Frozen** — 27 tok, FVE 0.642:
```
List of articles pattern
Political commentary tone
Final section header completes
```
**Co-train** — 25 tok, FVE 0.245:
```
newsletter format

political analysis content

"Looking deeper at"
```

### idx 440

**Source** (last 300 chars): …e construction of the Frecheirinha dam The project includes building a reservoir on the Caiçara River with a storage capacity of up to 85 million cubic meters of water. The stored water will be used for irrigation of agricultural land and water supply in the Coreaú River basin. - Feasibility studies

**Frozen** — 30 tok, FVE 0.719:
```
Project description format

technical services listed

"feasibility studies" needs elaboration
```
**Co-train** — 27 tok, FVE 0.827:
```
infrastructure project

contract scope list

"Feasibility studies"
```

### idx 460

**Source** (last 300 chars): …cay modes (bound beta decay of ‘bare’ or fully ionized nuclei), laser transmutation, nuclear forensics, radiation hormesis and the LNT hypothesis are covered. Atomic mass data for over 3000 nuclides from the most recent (2003) evaluation are included. The book contains a CD-ROM with the Universal Nu

**Frozen** — 42 tok, FVE 0.116:
```
technical nuclear physics description

continuation of nuclear data system context

"Next Nu" begins "Nuclide Chart" or "Nuclide Table"
```
**Co-train** — 25 tok, FVE 0.081:
```
nuclear physics content

technical documentation

"Interactive Nu"
```

### idx 480

**Source** (last 300 chars): …n gradient of copper-binding ligands across the sediment-water interface is much larger at site M than at site S, suggesting the stronger possibility of a benthic flux of such ligands from site M sediments. These differences may result from the dissimilar physical and chemical characteristics of the

**Frozen** — 30 tok, FVE 0.658:
```
Comparative sediment analysis
different particle characteristics
"of the" requires noun phrase
```
**Co-train** — 28 tok, FVE 0.708:
```
sediment characteristics

physical properties

"varied physical properties of the"
```

### idx 500

**Source** (last 300 chars): …y proteins, which favors the fixation of natural dyes in its structure, resulting in very bright and vivid colors. Its properties are high resistance, good moisture absorption and soft and smooth touch, with a very particular slight shine. It is one of the most elegant fibers and has been used since

**Frozen** — 32 tok, FVE 0.533:
```
product description format
historical/cultural context expected
"since" requires a time reference
```
**Co-train** — 26 tok, FVE 0.513:
```
product description

historical context

"has been used since"
```

### idx 520

**Source** (last 300 chars): … is the preferred replacement of niche market dielectric heat transfer applications using CFC-113. Based on the inert properties and high dielectric strength (>15kV/mm) PFC liquids have also been specified also in international military standards for radar cooling. Exceedingly stable, PFCs have been

**Frozen** — 32 tok, FVE 0.621:
```
Technical product description

Historical/material property context

"have been" requires past participle
```
**Co-train** — 23 tok, FVE 0.626:
```
technical brochure

chemical products

"have been"
```

### idx 540

**Source** (last 300 chars): …9a; COSEWIC, 2004). However, genetic analyses suggest that these two groups have been separate for approximately 8000 years (de March et al., 2002). The St. Lawrence belugas are also geographically isolated from other populations in the eastern Arctic, even though the distance that separates them is

**Frozen** — 33 tok, FVE 0.605:
```
Scientific comparison structure

distance between regions

"though the distance between them is" requires completion
```
**Co-train** — 29 tok, FVE 0.620:
```
biological description

ecological contrast

"though the distance between them is"
```

### idx 560

**Source** (last 300 chars): …r standard devices, custom voltage levels (-2.0 to 5.5 V) for proprietary devices, or compatibility with the latest LVDS digital electronics. With the deep onboard memory, you can link and loop complex patterns using the onboard FPGA hardware. You can synchronize NI high-speed digital devices at the

**Frozen** — 32 tok, FVE 0.751:
```
Technical documentation tone

Hardware synchronization context

"synchronize at the" requires a technical specification term
```
**Co-train** — 27 tok, FVE 0.820:
```
technical product description

precision timing capabilities

"synchronize at the"
```

### idx 580

**Source** (last 300 chars): …ons. Persons and groups engaging in First Amendment related activities have the legal right to: - Organize and participate in peaceful assemblies, including demonstrations, rallies, parades, marches, or other similar gatherings. - Conduct these assemblies or gatherings in a designated public area. -

**Frozen** — 29 tok, FVE 0.535:
```
Legal rights enumeration

Protest rights list continuation

"-" requires another right
```
**Co-train** — 27 tok, FVE 0.361:
```
legal rights enumeration

civil rights protest context

"- The right to"
```

### idx 600

**Source** (last 300 chars): …bility in the presence of water contamination. Good anti-foam and air release properties ensure smooth operation and system efficiency. - Rando® MV is a multiviscosity, anti-wear hydraulic fluid, designed for use in hydraulic systems subjected to wide variations in ambient and operating temperatures

**Frozen** — 30 tok, FVE 0.293:
```
Product specification format

Technical chemical description

"under these temperature conditions" closes description
```
**Co-train** — 25 tok, FVE -0.227:
```
product description

technical specifications

"exposed to elevated"
```

### idx 620

**Source** (last 300 chars): …ction has started for most devices, which will be used as baseline in the project effectiveness evaluation stage. - Deployment of services and use cases finalisation. - Development of testing protocol and monitoring through the platform, - Testing of use cases and KPI assessment. WP5 – Demonstration

**Frozen** — 35 tok, FVE 0.413:
```
Research project sections listed
"Demonstration" section expected
"Demonstration" begins a sub-section title
```
**Co-train** — 24 tok, FVE 0.213:
```
workshop sections

list structure

"Demonstration"
```

### idx 640

**Source** (last 300 chars): …ants with raw seam details. Side slash pockets; back welt pockets. Natural rise. Flared legs. Hook-and-eye/zip front. Cotton/viscose/spandex. Imported. Belted Relic Pants Product Info A brilliant tailored silk pant by Helmut Lang, cropped at the ankle with an adjustable button belt. Color: White Dry

**Frozen** — 30 tok, FVE 0.638:
```
product description format

fabric/material specifications

"Dry" likely begins "Dry Clean"
```
**Co-train** — 23 tok, FVE 0.232:
```
product description

fabric/material details

"dry"
```

### idx 660

**Source** (last 300 chars): …r is a good carrier of information once becoming infused with the new frequency. Our feelings and emotions are our driving force for making decisions for the most part and also define the way we behave. Every feeling and emotion has it’s own specific vibrational frequency. There is a flower, crystal

**Frozen** — 30 tok, FVE 0.351:
```
product listing pattern
nature/wellness themes
"crystal" requires continuation
```
**Co-train** — 28 tok, FVE 0.382:
```
product description

energy/lucky items list

"each item, crystal"
```

### idx 680

**Source** (last 300 chars): …turn dark purple in summer, and bright red in fall. Dark fruits are produced in summer, which are eaten by songbirds. This tough tree reaches 20 feet at maturity and grows best in full sun and well-drained soils. It is also noted for its tolerance to urban growing conditions. Prairiefire Crabapple (

**Frozen** — 36 tok, FVE 0.224:
```
Garden plant guide format

species list continuation

'Pyrus calleryana' opening parenthetical
```
**Co-train** — 28 tok, FVE 0.720:
```
plant list continuation

gardening blog format

"Crabapple ("
```

### idx 700

**Source** (last 300 chars): … the simulation than is the case with Λ cold dark matter (ΛCDM) because of the non-linear nature of the fifth force. Linear perturbation theory is shown to be a poor approximation for the f(R) models, except when the chameleon effect is very weak. We show that the relative differences from the fiduc

**Frozen** — 39 tok, FVE 0.352:
```
Scientific paper structure

technical cosmological analysis

"and a fiduc" begins "fiducial" cosmology parameter
```
**Co-train** — 27 tok, FVE 0.402:
```
scientific paper

data description

"as compared to the fiduc"
```

### idx 720

**Source** (last 300 chars): …l and the lightweight titanium frame. The Barrett M82, standardized by the U.S. military as the M107, is a recoil-operated, semi-automatic anti-materiel rifle developed by the American Barrett Firearms Manufacturing company. Despite its designation as an anti-materiel rifle, it is used by some armed

**Frozen** — 27 tok, FVE 0.083:
```
Wiki article format
weapon description continuation
"armed" requires completion
```
**Co-train** — 24 tok, FVE 0.263:
```
Wikipedia description

weapon article context

"armed"
```

### idx 740

**Source** (last 300 chars): …ther players anyway. Powers and Abilities Some classic ‘magic-type’ powers do not work well in many LARP situations. Powers that would require a player to pretend he doesn’t know something he knows are often off-limits: e.g. invisibly listening to conversations, mind-reading, and the like. Note that

**Frozen** — 33 tok, FVE 0.750:
```
Role-playing game rules context

note that introduces qualification

"note that" requires continuation explaining exception
```
**Co-train** — 23 tok, FVE 0.740:
```
rules list

exceptions coming

"Note that"
```

### idx 760

**Source** (last 300 chars): …arrival in the UK" which had not been "expressly waived", meaning she "had immunity at the time of Harry's death". Reacting to the decision, Harry's mum Charlotte said: "The governments and Mrs Sacoolas need to understand that this Court ruling is just a blip along the way, I promised my boy I would

**Frozen** — 34 tok, FVE 0.328:
```
Legal/accountability narrative
promised emotional testimony

"and I would" requires completion of a vow
```
**Co-train** — 25 tok, FVE 0.237:
```
obituary tribute

father's promise

"I would"
```

### idx 780

**Source** (last 300 chars): …e level of the school. These include changes in the content and organization of individual courses, the addition of new courses to the program or the substitution of one course for another as meeting a specific program requirement. Some, however, are significant enough to require the approval of the

**Frozen** — 33 tok, FVE 0.450:
```
Academic policy document structure

Explanatory clause continuation

"the" requires a specific noun
```
**Co-train** — 25 tok, FVE 0.687:
```
academic policy

committee authority

"requires approval of the"
```

### idx 800

**Source** (last 300 chars): …ngles in Taranaki today with our free online personals and free Taranaki chat! Taranaki is full of single men and women like you looking for dates, lovers, friendship, and fun. We have great pleasure in welcoming you to the 15th Taranaki Fringe Garden Festival. The Taranaki Fringe Garden Festival is

**Frozen** — 30 tok, FVE 0.618:
```
event description continuing
organizational details expected
"the Festival is" needs completion
```
**Co-train** — 27 tok, FVE 0.755:
```
event description

organizing body

"The Sunshine Coast Flower Show is"
```

### idx 820

**Source** (last 300 chars): …omposite frames under a column removal scenario, Journal of Building Engineering, Volume 44, December 2021, 102957. Kai Qian, Numerical evaluation of the reliability of using single-story substructures to study progressive collapse behaviour of multi-story RC frames, Journal of Building Engineering,

**Frozen** — 26 tok, FVE -1.069:
```
citation list pattern

journal citation structure

"Journal of Environmental"
```
**Co-train** — 25 tok, FVE 0.440:
```
reference list

journal citation

"Journal of Cleaner Production,"
```

### idx 840

**Source** (last 300 chars): … participation of more than a hundred active members. One of the most diverse groups in the AAR, the section’s presenters, panelists, and audience represent scholars at all stages of their academic careers. The section also features regular attendance and participation of international scholars from

**Frozen** — 32 tok, FVE 0.591:
```
Academic book description pattern

Geographic scope expansion expected

"from" requires destination countries
```
**Co-train** — 26 tok, FVE 0.537:
```
academic text

scholarly introduction

"lecturers from"
```

### idx 860

**Source** (last 300 chars): …rug ikea adum jute rugs large area under runner round lime green carpets and cape town inexpensive shaggy for living room baby light blue nautical black classroom burnt orange Final Thoughts On Choosing A Round Area Rug. A round area carpet is a attractive and functional floor

**Frozen** — 33 tok, FVE 0.709:
```
Decorative floor covering description

furniture store product description

"great floor" continues noun phrase
```
**Co-train** — 25 tok, FVE 0.663:
```
floor decoration guide

home improvement context

"a versatile floor"
```

### idx 880

**Source** (last 300 chars): …easures each fiber in microns) and represents the average diameter of the fiber sample measured in microns. It is also possible to sort and grade fibers without this test. The professionals who do this are able to sort raw fiber into ranges of only 3 microns at the rate of 150-200 pounds a day! That

**Frozen** — 32 tok, FVE 0.475:
```
practical example illustrating speed
factual/statistical elaboration
"That" closes relative clause
```
**Co-train** — 27 tok, FVE 0.330:
```
puzzle/number sequence

fascinating math revelation

"That"
```

### idx 900

**Source** (last 300 chars): …mportant. The quality of the images can enactment how your visitors perceive your website whether it is credible or not. Whether this is fair or not, its comprehensibly human psychology. We all tend to gravitate towards nice and pretty pictures, and are quick to regard as being considering something

**Frozen** — 31 tok, FVE 0.374:
```
marketing blog tone

credibility/supplement review context

"thinking something" incomplete
```
**Co-train** — 27 tok, FVE 0.469:
```
brand credibility advice

design principle pattern

"making you think something"
```

### idx 920

**Source** (last 300 chars): …Automation24, Inc. Datasheets for Optical Linear Encoders Optical linear encoders use fiber optic technology to sense position and displacement. Optical Linear Encoders: Learn more |Optical level sensor O1D300 from ifm electronic has a background suppression > 10-100 m and two programmable switching

**Frozen** — 35 tok, FVE 0.529:
```
Product specification listing

technical documentation pattern

"two switching" requires completion, likely "outputs" or similar
```
**Co-train** — 25 tok, FVE 0.464:
```
technical product description

features list pattern

"two switching"
```

### idx 940

**Source** (last 300 chars): …d to oem as oem are twice the cost?. 10 yz450. rebuild? - yz400f/426f/450f/450fx, I am mechanically inclined and do all of my own work, i can have the t. chain replaced in 30/35 min at a cost of $23. i just replaced the top end in my '09 yz450f after 110 hrs, i took the cylinder off and noticed that

**Frozen** — 32 tok, FVE 0.700:
```
Troubleshooting engine issue

description of mechanical inspection

"found that" introduces observed damage
```
**Co-train** — 24 tok, FVE 0.704:
```
bike mechanic narrative

inspection findings

"noticed that"
```

### idx 960

**Source** (last 300 chars): …eporting and his creative use of pop language and explosive punctuation. "As a [writer] of flamboyance he is without peer in the Western world," Joseph Epstein wrote in The New Republic. "His prose style is normally shotgun baroque, sometimes edging over into machine-gun rococo, as in his article on

**Frozen** — 37 tok, FVE 0.573:
```
Literary analysis of Capote's work

Specific book reference in progress

"essay on" requires a topic noun
```
**Co-train** — 25 tok, FVE 0.310:
```
biographical narrative

factual description

"essay on"
```

### idx 980

**Source** (last 300 chars): … In the bottom left corner, you will notice an “I” button where all the payouts are revealed. On top of that, all the options can be found there as well. Here are the highest paylines that each symbol line can offer. Play Caishen’s Arrival and you will understand them all really quickly. Pot of Gold

**Frozen** — 33 tok, FVE 0.527:
```
Online casino game list pattern

slot feature descriptions expected

"Pot of Gold" introduces slot title
```
**Co-train** — 28 tok, FVE 0.398:
```
poker game description

prize structure explanation

"Pot of Gold"
```

