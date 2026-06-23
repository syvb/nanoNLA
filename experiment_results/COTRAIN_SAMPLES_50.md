# Co-train vs frozen — 50 full held-out samples (λ=0.015)

Matched by `idx` on the **same** held-out documents (rows 25k–26k, doc-disjoint from RL training). Both explanations are single temperature-1.0 samples. FVE shown is on the **frozen base-AR ruler** (apples-to-apples per sample); the co-trained *system* number (own-AR ruler) is higher in aggregate — see [`COTRAIN_COMPARISON.md`](COTRAIN_COMPARISON.md).

- **frozen** = AV trained with a frozen critic (original sweep), FVE 0.450 / 25.2 tok.
- **co-train** = AV + AR co-trained (`--train-critic`), FVE 0.375 base-AR / 0.471 own-AR / 22.5 tok.

Pattern to watch: co-train drops the **genre/document-type framing line** ("scientific explanation", "political speech") in favour of the bare topic, and cuts the quoted-continuation snippet first.

---

### idx 0

**Source** (last 300 chars): …, Stacy Clark. Clark is a continuum navigator in Tabitha’s central and western regions, which includes the cities of Grand Island and York, Nebraska. Clark joined Tabitha over 15 years ago, serving as an expert and guiding clients and families through the overwhelming process of Senior Care services

**Frozen** — 28 tok, FVE 0.591:
```
community organization
support services
"educating families about behavioral health services"
```
**Co-train** — 23 tok, FVE 0.540:
```
Healthcare services

Patient experience

"services"
```

### idx 20

**Source** (last 300 chars): … to February 28th. Second, in March, VIP members will receive a free miniature Star Wars Republic Frigate with purchases of $50 or more. Third, free shipping is available for S@H orders from March 1st to March 15th. And finally, VIP members will receive double VIP points between March 25th and April

**Frozen** — 28 tok, FVE -0.075:
```
loyalty program details
membership tier
"becomes a" incomplete
```
**Co-train** — 23 tok, FVE -0.157:
```
Rewards program

"becomes a member"
```

### idx 40

**Source** (last 300 chars): … important for understanding future changes to ocean productivity brought about by a warming climate and changing environmental conditions.| |Rights:||All UHM dissertations and theses are protected by copyright. They may be viewed from this source for any purpose, but reproduction or distribution in

**Frozen** — 23 tok, FVE 0.071:
```
academic repository
distribution rights
"distribution in"
```
**Co-train** — 21 tok, FVE -0.301:
```
Copyright notice

distribution

"in"
```

### idx 60

**Source** (last 300 chars): …se Geometric Doormat This product is currently sold out. Strong fibers made from the husks of coconuts set the tone for a sustainable home. This durable doormat features a bold geometric pattern in a versatile muted turquoise color making it the perfect addition to any front door. - 18 x 30 in - 100

**Frozen** — 24 tok, FVE 0.682:
```
product description
material composition
"100"
```
**Co-train** — 24 tok, FVE 0.643:
```
Product description

Material composition

"100"
```

### idx 80

**Source** (last 300 chars): …much better overall.” The Hawkeyes struck first at the 9:05 mark as Scraper scored to give Iowa a 1-0 lead. In the 17th minute, Scraper scored off a rebound to push the Hawkeye lead to 2-0. Sarah Drake tallied a goal at the 42:28 mark to push the lead to 4-0. Moments later, Scraper tallied her third

**Frozen** — 22 tok, FVE 0.414:
```
score recap
player highlights
"third"
```
**Co-train** — 22 tok, FVE 0.362:
```
Game recap

scoring

"third"
```

### idx 100

**Source** (last 300 chars): …mmon to find clothing fiber, dirt and construction materials in computer facilities. Less common dust sources also exist. Natural events, such as volcanic eruptions and dust storms are some of the largest dust producing events possible. These events introduce a variety of different constituents into

**Frozen** — 23 tok, FVE 0.607:
```
airborne contaminants
particle dispersion
"into"
```
**Co-train** — 23 tok, FVE 0.597:
```
Airborne particles

contamination

"into"
```

### idx 120

**Source** (last 300 chars): …way we perceive, the way we sense this life. The moonlight is playing a very important role. During the day we have sunlight. The sunlight brings information into our body. During the night, during the dark period, the information is digested. But undigested information is stored in our body. During

**Frozen** — 24 tok, FVE 0.715:
```
spiritual explanation
moon's influence
"During"
```
**Co-train** — 25 tok, FVE 0.557:
```
Astronomical timing

sleep cycles

"During"
```

### idx 140

**Source** (last 300 chars): … no slot games rival fafafa™ gold progressives and levels. Will enjoy specific slots. Even better, you can buy specific totally free local casino dollars. If you are after real casino slots online that are incredibly quick and easy, the fafafa slot may well be the one for you. It's available at many

**Frozen** — 24 tok, FVE 0.800:
```
online casino review
game availability
"at many"
```
**Co-train** — 23 tok, FVE 0.771:
```
Online casinos

slot games

"at many"
```

### idx 160

**Source** (last 300 chars): …purchase of clothing as well. If you are using coupon codes for clothing to shop online, look for free shipping offers to save even more. Free Express US Shipping on $100+ Enter code at checkout to save! BOGO Coupon - Buy One Get One 50% Off Everything Entrer code at checkout for savings. Save Up to

**Frozen** — 25 tok, FVE 0.640:
```
promotional codes
discount offers
"Save Up to"
```
**Co-train** — 24 tok, FVE 0.614:
```
Promotions

Discounts

"Save Up to"
```

### idx 180

**Source** (last 300 chars): …. Walking performance was measured using the timed 25-ft walk (T25FW) and six-minute walk (6MW). Gait parameters were captured using a GaitRite™ electronic walkway. RESULTS: Aerobic capacity, balance, and knee-extensor asymmetry were associated with walking performance and gait in persons with MS (r

**Frozen** — 24 tok, FVE 0.315:
```
correlation analysis
statistical results
"(r"
```
**Co-train** — 20 tok, FVE -0.012:
```
Correlation

statistics

(r
```

### idx 200

**Source** (last 300 chars): …lian summers. After successful international expansions in Dubai, Doha, Cannes, Istanbul and, most recently, Egypt’s North Coast, Scalini has opened in Bahrain, in association with Sevens Holding, the fastest growing luxury hospitality company in the GCC region. The restaurant is located at the five

**Frozen** — 22 tok, FVE 0.414:
```
hotel description
"five"
"five"
```
**Co-train** — 23 tok, FVE 0.597:
```
Hotel description

five-star

"the five"
```

### idx 220

**Source** (last 300 chars): …sure, however, can come from all types of radioactive materials if they are inside the body. Once inside, much of the radiation energy will get absorbed in. Radiation Safety Training This course covers two main areas, the awareness of the safety problem and the legal framework. Firstly the awareness

**Frozen** — 25 tok, FVE 0.421:
```
training course description
continuing content
"the awareness"
```
**Co-train** — 22 tok, FVE 0.475:
```
Training content

Hazards

"The awareness"
```

### idx 240

**Source** (last 300 chars): … would be convincing shippers to spend the extra time coming upriver, and then, also convincing freight rail shippers to add the port as a regular scheduled stop so cargo could move quickly to markets west or north. "It is a bit of a chicken and egg problem," Hendrick said. Bnearing@nycap.rr.com • 5

**Frozen** — 22 tok, FVE 0.643:
```
contact information
phone number
"5"
```
**Co-train** — 21 tok, FVE 0.365:
```
Contact info

Phone

"5"
```

### idx 260

**Source** (last 300 chars): …us of DFS uncertain, and highlights separate legal issues in the potentially even more lucrative mobile sports betting market in New York. DFS may no longer be legal, but more traditional in-person sports betting on the outcome of actual sporting events is currently legal and operational at New York

**Frozen** — 25 tok, FVE 0.312:
```
gaming news report
legal issues
"New York"
```
**Co-train** — 20 tok, FVE 0.130:
```
Casino industry

New York
```

### idx 280

**Source** (last 300 chars): …annel, cast iron, round bar, square bar, water pipe, gas pipe, plumbing pipe, STK steel pipe, stainless - Abrasive Grain - Grain Size * Products that meet your criteria are shown in Customers Who Viewed This Item Also Viewed Sumitomo Electric Industries | 5 star | 4 star | 3 star | 2 star | 1 star 3

**Frozen** — 22 tok, FVE 0.441:
```
product review
rating system
"3"
```
**Co-train** — 21 tok, FVE 0.428:
```
Product reviews

Rating

"3"
```

### idx 300

**Source** (last 300 chars): …ttery for a runtime of 60 minutes. The X2L LED Series easily mounts to most subcompact pistol rails utilizing Insight's patented Slide-Lock interface. Replacement LED kits will be available later this year. For more information, click here Request information about this product Other Recent Products

**Frozen** — 24 tok, FVE 0.311:
```
product listing
recent items
"Other Recent Products"
```
**Co-train** — 24 tok, FVE 0.380:
```
Product listing

Related items

"Other Recent Products"
```

### idx 320

**Source** (last 300 chars): …I am sitting here feeling very sad over the LODD of Ray Barrett who was killed when the truck he was responding to a motor vehicle crash in rolled off the road yesterday just north of Hicks Bay at Waihau Bay. Of even more concern for

**Frozen** — 26 tok, FVE 0.484:
```
personal concern
firefighters' safety
"special concern for"
```
**Co-train** — 22 tok, FVE 0.449:
```
Concern

Specific

"Special concern for"
```

### idx 340

**Source** (last 300 chars): …s that we saw last year but actually below the long-term normal," he said. "The water into the landscape, the supply has really fallen off here over the last couple months." Rising lake levels have posed problems for some Michigan marinas and have threatened the loss of lakefront property during the

**Frozen** — 24 tok, FVE 0.650:
```
weather damage
real estate impact
"during the"
```
**Co-train** — 22 tok, FVE 0.635:
```
Flooding

damage

"during the"
```

### idx 360

**Source** (last 300 chars): … to: - Increase your revenue and access new markets due to universal compliance and certification results (e.g. European and International Cybersecurity Standards, GDPR, Eurosmart IoT Cybersecurity Certification, CC, CSPN, Eurosmart Iot SCS, ISA/IEC 62443, ARM DSA, SESIP, IoT SF, etc.) - Reduce your

**Frozen** — 24 tok, FVE 0.564:
```
risk management benefits
cost reduction
"reduce your"
```
**Co-train** — 21 tok, FVE 0.467:
```
Benefits

Risk

"reduce your"
```

### idx 380

**Source** (last 300 chars): …China. “The question of the trade imbalance is one that I think will be placed firmly on the table by the African delegates,” Sim Tshabalala, chief executive of South Africa’s Standard Bank Group, told reporters on Saturday. Critics also say Africa is loading itself up on Chinese debt that countries

**Frozen** — 24 tok, FVE 0.319:
```
Debt crisis
African countries
"countries"
```
**Co-train** — 19 tok, FVE 0.024:
```
Debt crisis

countries
```

### idx 400

**Source** (last 300 chars): …d||21/9/19 → 21/9/22| Bibliographical noteFunding Information: This work was supported by the National Research Foundation of Korea (NRF) funded by the Korea Government (Ministry of Science and ICT, MSIT) under Grant NRF-2020R1A2C3011697, and the Yonsei University Research Fund of 2021 (2021-22-0001

**Frozen** — 32 tok, FVE -0.570:
```
academic paper
funding acknowledgment
"2019R1A5A2"
```
**Co-train** — 18 tok, FVE -0.712:
```
Academic

Grant
```

### idx 420

**Source** (last 300 chars): …for sustainable growth. Support from unions and public-education advocates won Jim Kenney the primary election, despite $7 million in outside spending for his opponent. Activist Linda Sarsour on the Arab American struggle against police brutality. What a serious desegregation policy might look like.

**Frozen** — 25 tok, FVE 0.195:
```
editorial series
political commentary
"More on the"
```
**Co-train** — 18 tok, FVE 0.314:
```
News digest

policy
```

### idx 440

**Source** (last 300 chars): …e construction of the Frecheirinha dam The project includes building a reservoir on the Caiçara River with a storage capacity of up to 85 million cubic meters of water. The stored water will be used for irrigation of agricultural land and water supply in the Coreaú River basin. - Feasibility studies

**Frozen** — 25 tok, FVE 0.794:
```
project description
contract details
"Feasibility studies"
```
**Co-train** — 26 tok, FVE 0.763:
```
Project scope

Consulting services

"Feasibility studies"
```

### idx 460

**Source** (last 300 chars): …cay modes (bound beta decay of ‘bare’ or fully ionized nuclei), laser transmutation, nuclear forensics, radiation hormesis and the LNT hypothesis are covered. Atomic mass data for over 3000 nuclides from the most recent (2003) evaluation are included. The book contains a CD-ROM with the Universal Nu

**Frozen** — 24 tok, FVE 0.092:
```
academic presentation
radioactive decay
"Extended Nu"
```
**Co-train** — 21 tok, FVE 0.096:
```
Nuclear data

"Interactive Nu"
```

### idx 480

**Source** (last 300 chars): …n gradient of copper-binding ligands across the sediment-water interface is much larger at site M than at site S, suggesting the stronger possibility of a benthic flux of such ligands from site M sediments. These differences may result from the dissimilar physical and chemical characteristics of the

**Frozen** — 27 tok, FVE 0.711:
```
sediment types
geochemical differences
"characteristics of the"
```
**Co-train** — 25 tok, FVE 0.587:
```
Sediment cores

particle size

"of the"
```

### idx 500

**Source** (last 300 chars): …y proteins, which favors the fixation of natural dyes in its structure, resulting in very bright and vivid colors. Its properties are high resistance, good moisture absorption and soft and smooth touch, with a very particular slight shine. It is one of the most elegant fibers and has been used since

**Frozen** — 26 tok, FVE 0.504:
```
product description
historical context
"has been used since"
```
**Co-train** — 21 tok, FVE 0.510:
```
History

fibers

"since"
```

### idx 520

**Source** (last 300 chars): … is the preferred replacement of niche market dielectric heat transfer applications using CFC-113. Based on the inert properties and high dielectric strength (>15kV/mm) PFC liquids have also been specified also in international military standards for radar cooling. Exceedingly stable, PFCs have been

**Frozen** — 28 tok, FVE 0.615:
```
technical chemistry content
fluorinated hydrocarbons
"have been"
```
**Co-train** — 21 tok, FVE 0.535:
```
Materials

Safety

"have been"
```

### idx 540

**Source** (last 300 chars): …9a; COSEWIC, 2004). However, genetic analyses suggest that these two groups have been separate for approximately 8000 years (de March et al., 2002). The St. Lawrence belugas are also geographically isolated from other populations in the eastern Arctic, even though the distance that separates them is

**Frozen** — 26 tok, FVE 0.569:
```
biological distinction
geographical separation
"the distance is"
```
**Co-train** — 24 tok, FVE 0.305:
```
Geographic proximity

distance

"the distance is"
```

### idx 560

**Source** (last 300 chars): …r standard devices, custom voltage levels (-2.0 to 5.5 V) for proprietary devices, or compatibility with the latest LVDS digital electronics. With the deep onboard memory, you can link and loop complex patterns using the onboard FPGA hardware. You can synchronize NI high-speed digital devices at the

**Frozen** — 25 tok, FVE 0.725:
```
product description
technical precision
"synchronized at the"
```
**Co-train** — 20 tok, FVE 0.615:
```
Timing synchronization

"at the"
```

### idx 580

**Source** (last 300 chars): …ons. Persons and groups engaging in First Amendment related activities have the legal right to: - Organize and participate in peaceful assemblies, including demonstrations, rallies, parades, marches, or other similar gatherings. - Conduct these assemblies or gatherings in a designated public area. -

**Frozen** — 23 tok, FVE 0.461:
```
legal rights list
public assembly rights
"-"
```
**Co-train** — 21 tok, FVE 0.072:
```
Civil rights

Protest

"-"
```

### idx 600

**Source** (last 300 chars): …bility in the presence of water contamination. Good anti-foam and air release properties ensure smooth operation and system efficiency. - Rando® MV is a multiviscosity, anti-wear hydraulic fluid, designed for use in hydraulic systems subjected to wide variations in ambient and operating temperatures

**Frozen** — 26 tok, FVE 0.210:
```
technical data sheet
specification listing
"operating temperatures"
```
**Co-train** — 18 tok, FVE 0.119:
```
Product specs

technical
```

### idx 620

**Source** (last 300 chars): …ction has started for most devices, which will be used as baseline in the project effectiveness evaluation stage. - Deployment of services and use cases finalisation. - Development of testing protocol and monitoring through the platform, - Testing of use cases and KPI assessment. WP5 – Demonstration

**Frozen** — 26 tok, FVE 0.387:
```
project report sections
demonstration activities
"Demonstration"
```
**Co-train** — 23 tok, FVE 0.330:
```
Project sections

Demonstration

"Demonstration"
```

### idx 640

**Source** (last 300 chars): …ants with raw seam details. Side slash pockets; back welt pockets. Natural rise. Flared legs. Hook-and-eye/zip front. Cotton/viscose/spandex. Imported. Belted Relic Pants Product Info A brilliant tailored silk pant by Helmut Lang, cropped at the ankle with an adjustable button belt. Color: White Dry

**Frozen** — 22 tok, FVE 0.528:
```
product description
material listing
"Dry"
```
**Co-train** — 22 tok, FVE 0.607:
```
Product description

fabric composition

"Dry"
```

### idx 660

**Source** (last 300 chars): …r is a good carrier of information once becoming infused with the new frequency. Our feelings and emotions are our driving force for making decisions for the most part and also define the way we behave. Every feeling and emotion has it’s own specific vibrational frequency. There is a flower, crystal

**Frozen** — 25 tok, FVE 0.408:
```
energy healing content
listing of tools
"crystal"
```
**Co-train** — 24 tok, FVE 0.342:
```
Tarot

Crystals

"or a crystal"
```

### idx 680

**Source** (last 300 chars): …turn dark purple in summer, and bright red in fall. Dark fruits are produced in summer, which are eaten by songbirds. This tough tree reaches 20 feet at maturity and grows best in full sun and well-drained soils. It is also noted for its tolerance to urban growing conditions. Prairiefire Crabapple (

**Frozen** — 32 tok, FVE 0.245:
```
plant list continuation
flowering fruit trees
'Pyrus calleryana' opening
```
**Co-train** — 34 tok, FVE 0.302:
```
Apple trees

Hybrid names

'Pyrus calleryana 'Chanticleer' (
```

### idx 700

**Source** (last 300 chars): … the simulation than is the case with Λ cold dark matter (ΛCDM) because of the non-linear nature of the fifth force. Linear perturbation theory is shown to be a poor approximation for the f(R) models, except when the chameleon effect is very weak. We show that the relative differences from the fiduc

**Frozen** — 26 tok, FVE 0.409:
```
technical cosmology
parameter space
"at the fiduc"
```
**Co-train** — 24 tok, FVE 0.457:
```
Cosmology

parameters

"the fiduc"
```

### idx 720

**Source** (last 300 chars): …l and the lightweight titanium frame. The Barrett M82, standardized by the U.S. military as the M107, is a recoil-operated, semi-automatic anti-materiel rifle developed by the American Barrett Firearms Manufacturing company. Despite its designation as an anti-materiel rifle, it is used by some armed

**Frozen** — 25 tok, FVE 0.422:
```
military weapon description
armed forces context
"armed"
```
**Co-train** — 21 tok, FVE 0.285:
```
Weapons

military

"armed"
```

### idx 740

**Source** (last 300 chars): …ther players anyway. Powers and Abilities Some classic ‘magic-type’ powers do not work well in many LARP situations. Powers that would require a player to pretend he doesn’t know something he knows are often off-limits: e.g. invisibly listening to conversations, mind-reading, and the like. Note that

**Frozen** — 24 tok, FVE 0.720:
```
rule clarification
"note that"
"note that"
```
**Co-train** — 18 tok, FVE 0.538:
```
Rules

Note that
```

### idx 760

**Source** (last 300 chars): …arrival in the UK" which had not been "expressly waived", meaning she "had immunity at the time of Harry's death". Reacting to the decision, Harry's mum Charlotte said: "The governments and Mrs Sacoolas need to understand that this Court ruling is just a blip along the way, I promised my boy I would

**Frozen** — 23 tok, FVE 0.342:
```
tragic loss
personal vow
"I would"
```
**Co-train** — 23 tok, FVE 0.184:
```
Personal resolve

fight for justice

"I would"
```

### idx 780

**Source** (last 300 chars): …e level of the school. These include changes in the content and organization of individual courses, the addition of new courses to the program or the substitution of one course for another as meeting a specific program requirement. Some, however, are significant enough to require the approval of the

**Frozen** — 25 tok, FVE 0.650:
```
academic policy
approval process
"requires approval of the"
```
**Co-train** — 26 tok, FVE 0.402:
```
Policy document

approval process

"subject to approval of the"
```

### idx 800

**Source** (last 300 chars): …ngles in Taranaki today with our free online personals and free Taranaki chat! Taranaki is full of single men and women like you looking for dates, lovers, friendship, and fun. We have great pleasure in welcoming you to the 15th Taranaki Fringe Garden Festival. The Taranaki Fringe Garden Festival is

**Frozen** — 26 tok, FVE 0.769:
```
event description
organizational details
"The Fraser Island Festival is"
```
**Co-train** — 22 tok, FVE -0.041:
```
Event description

"South Australian"

"is"
```

### idx 820

**Source** (last 300 chars): …omposite frames under a column removal scenario, Journal of Building Engineering, Volume 44, December 2021, 102957. Kai Qian, Numerical evaluation of the reliability of using single-story substructures to study progressive collapse behaviour of multi-story RC frames, Journal of Building Engineering,

**Frozen** — 24 tok, FVE -1.112:
```
citation list
journal name
"Journal of Environmental"
```
**Co-train** — 24 tok, FVE 0.433:
```
Journal citations

references

"Journal of Cleaner Production,"
```

### idx 840

**Source** (last 300 chars): … participation of more than a hundred active members. One of the most diverse groups in the AAR, the section’s presenters, panelists, and audience represent scholars at all stages of their academic careers. The section also features regular attendance and participation of international scholars from

**Frozen** — 22 tok, FVE 0.627:
```
academic bibliography
international scholars
"from"
```
**Co-train** — 22 tok, FVE 0.487:
```
Scholarly

Islamic

countries from
```

### idx 860

**Source** (last 300 chars): …rug ikea adum jute rugs large area under runner round lime green carpets and cape town inexpensive shaggy for living room baby light blue nautical black classroom burnt orange Final Thoughts On Choosing A Round Area Rug. A round area carpet is a attractive and functional floor

**Frozen** — 24 tok, FVE 0.577:
```
home decoration advice
floor covering
"ideal floor"
```
**Co-train** — 18 tok, FVE 0.106:
```
Home decor

floor
```

### idx 880

**Source** (last 300 chars): …easures each fiber in microns) and represents the average diameter of the fiber sample measured in microns. It is also possible to sort and grade fibers without this test. The professionals who do this are able to sort raw fiber into ranges of only 3 microns at the rate of 150-200 pounds a day! That

**Frozen** — 23 tok, FVE 0.391:
```
fishing context
impressive number
"That"
```
**Co-train** — 20 tok, FVE 0.378:
```
Speed

Efficiency

"That"
```

### idx 900

**Source** (last 300 chars): …mportant. The quality of the images can enactment how your visitors perceive your website whether it is credible or not. Whether this is fair or not, its comprehensibly human psychology. We all tend to gravitate towards nice and pretty pictures, and are quick to regard as being considering something

**Frozen** — 23 tok, FVE 0.214:
```
website content
image caption
"thinking something"
```
**Co-train** — 24 tok, FVE 0.304:
```
Trust perception

Website credibility

"somethign"
```

### idx 920

**Source** (last 300 chars): …Automation24, Inc. Datasheets for Optical Linear Encoders Optical linear encoders use fiber optic technology to sense position and displacement. Optical Linear Encoders: Learn more |Optical level sensor O1D300 from ifm electronic has a background suppression > 10-100 m and two programmable switching

**Frozen** — 24 tok, FVE 0.438:
```
product description
technical spec list
"two switching"
```
**Co-train** — 19 tok, FVE 0.216:
```
Sensor specs

switching
```

### idx 940

**Source** (last 300 chars): …d to oem as oem are twice the cost?. 10 yz450. rebuild? - yz400f/426f/450f/450fx, I am mechanically inclined and do all of my own work, i can have the t. chain replaced in 30/35 min at a cost of $23. i just replaced the top end in my '09 yz450f after 110 hrs, i took the cylinder off and noticed that

**Frozen** — 23 tok, FVE 0.738:
```
engine rebuild
inspection finding
"noticed that"
```
**Co-train** — 23 tok, FVE 0.653:
```
Car repair

engine inspection

"noticed that"
```

### idx 960

**Source** (last 300 chars): …eporting and his creative use of pop language and explosive punctuation. "As a [writer] of flamboyance he is without peer in the Western world," Joseph Epstein wrote in The New Republic. "His prose style is normally shotgun baroque, sometimes edging over into machine-gun rococo, as in his article on

**Frozen** — 27 tok, FVE 0.571:
```
film review context
celebrity culture critique
"essay on"
```
**Co-train** — 24 tok, FVE 0.563:
```
Book review

cultural analysis

"essay on"
```

### idx 980

**Source** (last 300 chars): … In the bottom left corner, you will notice an “I” button where all the payouts are revealed. On top of that, all the options can be found there as well. Here are the highest paylines that each symbol line can offer. Play Caishen’s Arrival and you will understand them all really quickly. Pot of Gold

**Frozen** — 28 tok, FVE 0.502:
```
casino game listing
rewards/prizes
"Pot of Gold"
```
**Co-train** — 24 tok, FVE 0.556:
```
Slot features

Jackpot

"Pot of Gold"
```

