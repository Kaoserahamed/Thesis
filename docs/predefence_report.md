# Analyzing & Forecasting River Morphological Evolution Using Machine Learning & Spatiotemporal Neural Models: A Case Study on the Padma River

**Shahjalal University of Science and Technology**
**Department of Computer Science and Engineering**

- **Md. Kaoser Ahamed Anik** — Reg. No.: 2020331019
- **S. S. Mahmud Turza** — Reg. No.: 2020331039

**Supervisor:** Md. Shadmim Hasan Sifat, Lecturer, Department of CSE, SUST

*July 19, 2026*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Related Work](#3-related-work)
4. [Study Area and Dataset](#4-study-area-and-dataset)
5. [Methodology](#5-methodology)
6. [Spatiotemporal Statistical Analysis of Padma River Dynamics](#6-spatiotemporal-statistical-analysis-of-padma-river-dynamics)
7. [Experimental Results and Analysis](#7-experimental-results-and-analysis)
8. [Discussion](#8-discussion)
9. [Future Works and Research Directions](#9-future-works-and-research-directions)
10. [Conclusion](#10-conclusion)
11. [References](#references)

---

## Abstract

Rivers are dynamic geomorphological systems experiencing continuous transformation through erosion and sediment transport. In Bangladesh, these processes displace 50,000–200,000 people annually, causing severe losses along the Padma River according to Natural Resources Defense Council (NRDC). Traditional monitoring via field surveys and manual remote sensing proves costly, spatially limited, and unsuitable for multi-year prediction. This paper presents a comprehensive framework for forecasting river morphological evolution from freely available satellite data using spatiotemporal deep learning. A 38-year time series (1987–2025) of binary water masks was constructed from Landsat (5 TM, 7 ETM+, 8 OLI) and Sentinel (Sentinel-2 MSI, Sentinel-1 SAR) imagery via Google Earth Engine. Water extraction used Modified Normalised Difference Water Index (MNDWI). The proposed Bidirectional ConvLSTM gap-filling model achieved IoU = 0.7366, outperforming classical methods. Five spatiotemporal architectures (ConvLSTM, U-Net+LSTM, Attention U-Net+ConvLSTM, Swin Transformer, ViT-based model) were evaluated across yearly, quarterly, and bi-monthly resolutions. Hybrid CNN-LSTM models consistently outperformed pure transformers. Attention U-Net+ConvLSTM achieved best yearly performance (IoU = 0.7005); U-Net+LSTM led bi-monthly prediction (IoU = 0.7791). Statistical analysis quantified mean annual migration of 255.4 m yr⁻¹ with 1998 extreme of 1,485.5 m yr⁻¹. An expansion of about 290 km² was projected, and a spatially explicit risk map was generated through long-term forecasting (2026–2040). Results demonstrate that freely available satellite imagery with deep learning can provide a practical, scalable framework for riverbank hazard monitoring and disaster management.

**Keywords:** River morphology, Riverbank erosion forecasting, Spatiotemporal neural networks, ConvLSTM, U-Net, Attention mechanisms, Satellite remote sensing, MNDWI, Time series prediction

---

## Acknowledgements

In this regard, we would like to thank all the people who have helped us to successfully complete this thesis work.

First of all, we would like to thank our supervisor Md. Shadmim Hasan Sifat, Lecturer, Department of Computer Science and Engineering, Shahjalal University of Science and Technology for his valuable guidelines and constructive feedbacks.

We would like to thank the Department of Computer Science and Engineering, SUST for the availability of the facilities in conducting this research work. Moreover, we are thankful to all other faculties whose valuable suggestions were helpful for completing this thesis work at different stages.

We are thankful to all open-source satellite imagery suppliers for their help; especially, we are grateful to United States Geological Survey for Landsat images and European Space Agency for Sentinel images. These sources have played an important role in our research.

**Md. Kaoser Ahamed Anik**
**S. S. Mahmud Turza**
*November 2025*

---

## List of Tables

| Table | Caption |
|-------|---------|
| 4.1 | Summary of image availability and quality across different temporal resolutions. |
| 4.2 | Performance comparison of gap-filling methods. |
| 5.1 | Data splitting strategy. |
| 6.1 | Decadal erosion, accretion, and net channel change. |
| 7.1 | ConvLSTM Performance on Yearly Data. |
| 7.2 | U-Net+LSTM Performance on Yearly Data. |
| 7.3 | Attention U-Net+ConvLSTM Performance on Yearly Data. |
| 7.4 | Vision Transformer Performance on Yearly Data. |
| 7.5 | Swin Transformer Performance on Yearly Data. |
| 7.6 | Detailed Year-wise Prediction Results (10-Year Prediction: 2016–2025). |
| 7.7 | Detailed Year-wise Prediction Results (5-Year Prediction: 2021–2025). |
| 7.8 | ConvLSTM Performance on Quarterly Data (Best Sequence per Setup). |
| 7.9 | U-Net+LSTM Performance on Quarterly Data (Best Sequence per Setup). |
| 7.10 | Attention U-Net+ConvLSTM Performance on Quarterly Data (Best Sequence per Setup). |
| 7.11 | Vision Transformer (ViViT) Performance on Quarterly Data (Sequence Length = 6). |
| 7.12 | Swin Transformer Performance on Quarterly Data (Best Sequence per Setup). |
| 7.13 | Detailed Quarterly Prediction Results — Setup 1 (Attention U-Net+ConvLSTM, SeqLen=8). |
| 7.14 | Detailed Quarterly Prediction Results — Setup 2 (Attention U-Net+ConvLSTM, SeqLen=10). |
| 7.15 | ConvLSTM Performance on Bi-Monthly Data. |
| 7.16 | U-Net+LSTM Performance on Bi-Monthly Data. |
| 7.17 | Attention U-Net+ConvLSTM Performance on Bi-Monthly Data. |
| 7.18 | SwinST Performance on Bi-Monthly Data. |
| 7.19 | Vision Transformer + Temporal Transformer Performance on Bi-Monthly Data. |

## List of Figures

| Figure | Caption |
|--------|---------|
| 4.1 | Study area along the mid-course of the Padma River. |
| 4.2 | Two representative examples of partially missing. |
| 5.1 | ConvLSTM Architecture. |
| 5.2 | U-Net + ConvLSTM Architecture. |
| 5.3 | Attention U-Net + ConvLSTM Architecture. |
| 5.4 | Swin Spatio-Temporal Transformer architecture with a shared encoder, temporal fusion module, and U-Net style decoder. |
| 5.5 | ViT-based spatio-temporal architecture with per-frame spatial tokenization, temporal transformer fusion, and convolutional decoder. |
| 6.1 | Long-term annual water area. |
| 6.2 | Seasonal water area dynamics of the Padma River (1988–2025). |
| 6.3 | Quarterly heat-map of water area. |
| 6.4 | Annual and decadal erosion and accretion of the channel. |
| 6.5 | Annual lateral migration rate of the centreline. |
| 7.1 | Long-term combined morphological risk for 2026–2040. |

---

# 1. Introduction

## 1.1 Background and Context

Rivers are one of the most dynamic features of Earth's landscape, and continuous reshaping occurs by the processes of erosion, sediment transport, and deposition. Such morphological changes take place in various spatial and temporal scales, ranging from daily fluctuations during flood events to decadal shifts in channel position and form. Understanding and predicting these changes is fundamental for water resource management, safeguarding human settlements, maintaining agricultural productivity, and preserving ecosystem integrity.

Bangladesh, being surrounded by the world's largest delta system formed by the network of the Ganges-Brahmaputra-Meghna river systems, is facing very serious challenges caused by river dynamics [1]. The country's rivers are characterized by high sediment loads, seasonal flooding during monsoon periods, and rapid channel migration rates which could be more than several hundred meters during monsoon seasons [2, 3].

## 1.2 Problem Statement

One of the main environmental hazards to which people in Bangladesh are subjected is riverbank erosion. The problem displaces about 50,000 to 200,000 people every year and results in serious economic losses with respect to agricultural land losses, damage of infrastructure, and finally forced migration to other areas [4]. The socio-economic impacts go far beyond the issues of direct displacement to food security, access to education, and long-term livelihood resilience among vulnerable communities [5, 6]. Despite the several decades of satellite imagery available, these highly valued datasets are not yet serving their full potential in informing our understanding or predictions of dynamic river processes [7]. Current field-survey and manual-based approaches are extremely time-consuming, expensive, and can only be conducted over relatively small spatial areas or at infrequent repeat cycles. Classic remote sensing techniques mainly rely on a change-detection approach, comparing discrete instants in time and thus missing out on the temporal evolution and spatial dependencies that come intrinsically with river morphodynamics. Recent breakthroughs in deep learning show unparalleled success in the learning of sophisticated spatiotemporal patterns from sequential data [8, 9]. Yet, their application to river morphology forecasting is still relatively scarce. Besides, no sufficient systematic evaluation has been performed yet regarding different deep learning architectures to reveal which methods are the most efficient for a multi-year prediction of river morphology. Limited predictability of river morphological changes impairs effective decision-making on disaster preparedness, land use, infrastructure development, and climate adaptation strategies.

## 1.3 Research Objectives

The main aim of the research work will be to develop data-driven predictive models for riverbank locations and channel morphological forecasts using spatiotemporal neural networks. The research objectives are as follows:

1. Compile a comprehensive dataset of satellite images spanning close to four decades (1987–2025) and design effective water-masking algorithms based on spectral indices.
2. Design and test several architectures of deep learning (U-Net + ConvLSTM network, Attention U-Net + ConvLSTM network) able to take into consideration both spatial and temporal characteristics of river evolution.
3. Allow pre-emptive forecasts of river morphology so that sufficient time can be obtained to plan or act accordingly.
4. For model assessment, there can be a range of criteria using which the strengths, weaknesses, and applicability of different architectures can be identified.
5. Establish a framework that can easily be expanded to analysis of socio-economic impacts and risk mapping [10].

## 1.4 Significance and Contributions

The current research further expands both theoretical knowledge and applications in the area of river morphology forecasting. Scientifically, it represents an innovative application of spatiotemporal neural networks in multi-year river forecasting, providing a comprehensive comparison of architectures to derive empirical knowledge about the appropriateness of these networks in the area. The creation of a model with sequence-based learning, tailored to long-term river behavior, represents an advancement in geophysical forecasting. The ability to make accurate forecasts allows river bank erosion events to be predicted in advance. These forecasts can then be used by agricultural planners to plan land use in regions where erosion can occur. The development of roads and bridges can then be undertaken based on the knowledge of prime erosion danger zones. Generally, these tools can aid in adapting to climate change because they can help predict how rivers behave based on changes in climate.

---

# 2. Literature Review

Riverbank erosion monitoring has evolved with satellite remote sensing technology. Early studies used aerial photography and manually digitized river boundaries, very labor-intensive and offering only very limited temporal coverage. The advent of multi-decadal satellite archives, especially the Landsat missions, has revolutionized the field by allowing systematic long-term monitoring of morphological changes in rivers.

## 2.1 Optical Satellite-Based Monitoring

Binh et al. [7] developed the Sentinel-based River Bank Erosion Detection (SRBED) method to extract the position of riverbanks from Landsat time series. The authors' approach showed the potential benefits of satellite archives for long-term morphological change detection over a large area. Their method relied on automated water extraction techniques using spectral indices, hence reducing manual interpretation drastically. However, the main theme in their research was the detection of historical changes rather than predictive models. Indeed, many studies in this area exhibit such behavior.

Langhorst et al. [11] implemented an exhaustive analysis regarding global riverbank erosion and accretion using Landsat data, creating algorithms able to detect riverbank channel movements. Their output allows setting up baseline erosion rates in global river systems and highlighting global hotspots where such events happen at faster rates. It gives a global perspective to assess the level of severity related to erosion in individual global nations, like Bangladesh. But worth mentioning here is that the above research was not based on any ML algorithms to predict erosion.

Several studies have focused on the systematic observation of river dynamic changes through combined remote sensing and GIS analysis. For example, the benefits of multi-temporal analysis in studying the pattern of erosion and long-term channel migration trend analysis have been identified by Langat et al. [12]. For example, river channel migration analysis in Bangladesh was implemented using these analysis methods by Islam and Haque [13], and from then on, these studies formed the basics of further research in the region. Further spatiotemporal analysis with quantitative descriptions of rates of erosion and accretion was implemented by Raju et al. [14] based on these analysis basics, and the diversity in river dynamic changes, where erosion points alternate with stable or accretion points, was stressed.

## 2.2 Synthetic Aperture Radar Applications

Remote sensing using synthetic aperture radar was found to complement optical remote sensing in tropical environments where persisting clouds make optical data sparse in space and time. In fact, the study by Freihardt et al. [2] using temporal series of SAR data from Sentinel-1 satellites processed in Google Earth Engine effectively mapped riverbank erosion in the tropical region of Bangladesh. The data obtained gave clear evidence of the capability of SAR data in giving yearlong observation irrespective of the season or atmospheric condition or illumination. Erosion was found above 100 m/year in the monsoons in some sections of the riverbank in Bangladesh, which highlights the extent of riverbank erosion in Bangladesh. The technical aspects of the process using Google Engine Scripts in detail are found in supplementary documents in the research paper [15]. In fact, like many other studies involving remote sensing data, although descriptive in nature, the study didn't predict data using any model or algorithm from the domain of AI or ML.

## 2.3 Regional Case Studies and Applications

Several studies took into consideration specific river systems to understand local morphodynamic processes. Tha et al. [16] evaluated riverbank erosion hotspots along the Mekong River in Cambodia by using remote sensing and hazard exposure mapping. Their integrated approach brought together physical erosion analysis with vulnerability assessments of exposed populations and infrastructure, hence supporting the importance of connecting geophysical processes with socio-economic consequences.

Saha [17] performed multi-sensor analysis by integrating optical and radar data to provide comprehensive morphological characterization of riverbank migration and island dynamics at the Padma confluence. Complex erosion, deposition, and island formation patterns were documented, resulting from junction dynamics and seasonal variations in flow. Haque [18] carried out the change detection study of the Jamuna River and found that it caused significant changes in local land use and settlement with dramatic losses of agricultural land and displacement over multi-decadal timescales.

Ritu et al. [19] modeled the bank shifting of the Padma River along with the LULC using geospatial analysis. But, their dataset contained 10-year intervals between satellite images and relied mostly on data from the summer season; this created gaps in both temporal and seasonal aspects that weaken prediction accuracy. Their model also relies a lot on historical trends, that makes it unable to capture any sudden or irregular future changes.

Another study [20] performed multitemporal Landsat imagery and classification-based mapping for quantifying erosion and accretion, therefore adopting a systematic method for measuring morphological change without predictive modeling.

## 2.4 Advanced Segmentation and Risk Assessment

Rafat et al. [10] proposed a new satellite-based mapping and quantifying methodology for erosion-prone riverbanks and villages lost in Bangladesh by using Segment Anything Model (SAM), which is a foundation model developed for image segmentation. The work showed that recent advancements in computer vision have enhanced the erosion mapping accuracy, especially for identifying settlements affected by erosion. Though the method produced highly accurate segmentations, it was focused on historical mapping without forecasting future changes and did not incorporate temporal sequence modeling.

Various risk assessment frameworks have been developed focusing on the identification of vulnerable areas. Ren [21] developed a large-scale riverbank erosion risk assessment model driven by remote sensing inputs along with topographic hydrological and land cover variables. Spatial analysis techniques have been used in this model to delineate areas of high risk using environmental factors of susceptibility, although the model is mostly based on static predictor variables rather than dynamic temporal evolution.

---

# 3. Related Work

Recent deep learning advances have enabled the detection of water bodies and river channel delineation from satellite imagery that is considerably superior and largely supplants traditional threshold-based methods relying on empirical indices such as NDWI or MNDWI.

## 3.1 Semantic Segmentation Architectures

Yuan et al. [22] proposed deep learning-based multispectral satellite image segmentation for water body detection and showed that CNNs could achieve superior accuracy compared to the traditional methods. It worked effectively in scenes with mixed land cover, various shadows, or variable water turbidity, which challenged threshold-based techniques due to complications. These CNNs have been considered superior in recent years and are stated as the better approach for automated water extraction.

Li et al. [23] proposed an attention-enhanced multi-scale residual U-Net structure for segmenting water bodies from Sentinel-2 imagery. Their network architecture included an attention mechanism to focus on relevant features and suppress irrelevant information, together with a multi-scale mechanism to capture water bodies of different sizes. In particular, the attention enhancement proved very useful for discriminating water from other spectrally similar classes such as shadows or dark soil. This work directly informs our approach by demonstrating the value of attention mechanisms in boosting the accuracy of segmentations in complex situations.

Ghosh et al. [24] presented the following methods related to the extraction of water bodies from high-resolution remote sensing images using an upgraded U-Net model based on multi-scale information fusion. The reference explores architectural advances aimed at effectively using information at multiple spatial scales, thereby achieving enhanced detection performance and edge definition. In fact, it was found that the multi-scale approach was most beneficial in the case of rivers characterized by varying dimensions.

## 3.2 Cloud-Based Processing Platforms

The emergence of the cloud geospatial platform, including Google Earth Engine in particular, has increased the availability of satellite image processing services at large scales. Rapid and accurate flood mapping was found achievable using both Sentinel-1 and Landsat data in Google Earth Engine by DeVries et al. [25]. Scalability was identified as an important consideration in practical applications related to real-time monitoring in the above-noted research.

Clement et al. [26] demonstrated the mapping of flood dynamics in an unvegetated, ephemeral river using Google Earth Engine, thereby showcasing the capability of the tool in handling arduous environments with low vegetative cover and complex dynamic characteristics. Tripathy and Mallick [27] used Google Earth Engine in mapping and monitoring floods in the downstream parts of Mekong River, thereby proving its competency in handling larger-scale operations. Sazib et al. [28] mapped large-scale floods using the data from the Synthetic Aperture Radar satellite and analyzed the damage caused in the agricultural and population sectors in the Ganga & Brahmaputra River Basin, thereby combining both physical mapping and exposure analysis to help disaster management operations.

While these segmentation models excel at extracting water extent from individual images, they do not inherently capture temporal dynamics. They process each image independently without considering the sequential nature of river evolution or leveraging temporal patterns to improve predictions. This fundamental limitation drives the need for the implementation of recurrent architectures which have the capability to learn from temporal sequence and make predictions based on the past observations.

## 3.3 Recurrent Networks for Hydrological Forecasting

It has been demonstrated by research works that Long Short-Term Memory (LSTM) networks are effective in a number of tasks related to the prediction of hydrological time-series data and excel over traditional statistical methods in terms of performance. The application of LSTM networks in the prediction of river discharge was proposed by Fan et al. [8]. It was shown by the researchers how temporal relationships in hydrological data could be learned through the use of recurrent networks.

Le et al. [29] presented an integrated approach that combined convolutional neural networks (CNN) and long short-term memory network (LSTM) designs. The approach capitalized on the strengths of CNN in spatial feature extraction and the strengths of LSTM in temporal tasks, showing how the growing need for designs that combined the strengths of two network types was driving the increased adoption of hybrid designs.

Xiang et al. [30] explored the use of hybrid CNN-LSTM architectures in river flow forecasting, carrying out a comprehensive benchmarking of different architectural designs. A comparison can thus help gain an understanding of optimal designs that can leverage CNN and LSTM components in river flow forecasting effectively and outperform CNN-only and LSTM-only designs.

Horvath et al. [31] demonstrated the capability of LSTM networks in predicting water levels in the Tisza River in Central Europe. These studies show that the selection of variables entering the network and the duration of these variables are critical issues in the use of LSTM networks.

Li et al. [23] analyzed the performance and explainability of CNN-LSTM-Attention models in forecasting the daily streamflow in basins around the eastern Qinghai-Tibet Plateau. Their methodology helped in enhancing both performance and explainability using the attention mechanism in recognizing the most significant temporal inputs. It can thus be identified that attention contributes towards boosting the performance of recurrent models in spatiotemporal forecasting tasks.

## 3.4 ConvLSTM and Spatiotemporal Architectures

In comparison to conventional LSTMs, which process series of vectors, there exist many geophysical processes with strong spatial characteristics that need to be preserved in the course of temporal processing. To resolve these issues, convolutional LSTMs introduce convolution layers in place of matrix multiplications in LSTM cells in order to maintain spatial locality while processing temporal patterns.

Dehghani et al. [9] performed a comparative assessment of the performance of LSTM, CNN, and ConvLSTM in spatiotemporal forecasting tasks. Their comparison allowed the collection of empirical evidence about the characteristics of these architectures in comparison to different aspects of forecasting tasks. It was found that the performance of ConvLSTM was superior to other architectures in tasks involving spatiotemporal forecasting; nevertheless, ConvLSTM was computationally intensive. Their research highlights the reason for employing multiple architectures in the current analysis rather than using all architectures based on the performance of any individual model.

In fact, satellite image processing using deep learning algorithms was explored by Lemenkova [1] to track the flood dynamics in the Ganges Delta in Bangladesh. The issue takes direct relevance in the current discussion since it tackles the same geographical area and proves the applicability of combined convolutional and recurrent layers in handling spatial and temporal data related to the water extent. It was shown how complex patterns in flood dynamics can be identified using deep learning algorithms, although the task was focused on inundation mapping under isolated events, rather than morphological forecasting.

## 3.5 Socio-Economic Impacts of Riverbank Erosion

Insight into the human dimensions of riverbank erosion provides a better picture of why predictive modeling and early warnings are critical. The socioeconomic consequences of riverbank erosion and associated migration patterns have been studied by Islam et al. [4]. The authors identified quantitative proofs of displacement and economic losses, and riverbank erosion-related migration usually results in movement towards cities. As a consequence, this process causes quick urbanization, growth of informal settlements and such effects as interruptions in household earnings, continuity of education for children, and social network damages. It was shown that riverbank erosion had its human dimensions, which should be considered during risk management.

Das et al. [5] conducted case studies on different kinds of impacts caused by riverbank erosion in communities such as home loss, agricultural land, and infrastructure in the community. In doing so, the researchers revealed such impacts of riverbank erosion as psychological trauma due to displacement and the problem with reconstruction of their lives in other places.

Alam et al. [3] investigated the impacts and livelihood vulnerability of the riverbank erosion hazard in rural households near the Padma River. Extensive field surveys showed the consequences of the destruction of farmlands, homes, and communal infrastructure leading to food insecurity, educational accessibility issues, and negative health effects. Such research highlights the significant vulnerability of the poorest households due to their low adaptive capacity and dependence on lands for livelihood. Recovery paths may differ considerably depending on asset, social capital, and support system availability.

Sarkar et al. [6] analyzed riverbank erosion and livelihood resilience via traditional approaches in Northern Bangladesh taking into account indigenous coping measures and local knowledge systems. Such research highlighted the importance of the traditional approach. However, the traditional approach will have limitations in cases of accelerating erosion rates and changes in hydrological regimes.

The socio-economic research provides a more comprehensive picture of the need for a forecasting system providing sufficient lead time for communities to either adapt or resettle in a planned manner, rather than an emergency one. Human costs due to riverbank erosion become the main driving force behind the creation of prediction models for early warning systems.

## 3.6 Research Gaps

This review of literature and related studies identifies a number of key deficiencies that help to define the need for this particular research. Firstly, a large proportion of current research efforts have been devoted to the detection of changes from past events rather than making predictions about future river morphology. Although some research employs machine learning in susceptibility mapping, there is a lack of predictive modeling for future morphology. Second, although deep learning has been successfully applied to water body segmentation and streamflow prediction, its application to river morphology forecasting remains limited, with most studies continuing to rely on traditional change detection methods. Third, there is insufficient systematic comparison of different deep learning architectures for river morphology prediction tasks. Finally, most studies address either physical processes or social impacts in isolation, but rarely integrate both perspectives within a unified framework.

## 3.7 Novelty of our work

This research addresses these gaps by combining long-term satellite observations with spatiotemporal deep learning for multi-year river morphology forecasting. Our approach makes several key innovations: First, we develop sequence-based learning frameworks treating river evolution as continuous spatiotemporal processes using multi-year input windows to capture decadal trends. Second, we perform a systematic comparison between two architectures, namely U-Net with ConvLSTM and Attention U-Net with ConvLSTM, with empirical evidence for geomorphological prediction tasks. Third, we extend the prediction horizons from the usual short-term forecasts to multi-year predictions, which are crucial for infrastructure planning and disaster preparedness. Fourth, we adopt attention mechanisms to identify morphologically active zones automatically. Our framework utilizes freely available Landsat imagery through Google Earth Engine, hence accessible even for resource-constrained regions. Whereas most previous studies have addressed historical analysis or socio-economic impacts separately, our research lays the foundation for integrated systems linking physical predictions with comprehensive risk evaluation.

---

# 4. Study Area and Dataset

## 4.1 Study Area

*Figure 4.1: Study area along the mid-course of the Padma River.*

The area of interest (Figure 4.1) spans a large mid-course section of the Padma River, including its active channel, confluent or proximal floodplains, and large island/bar complexes. The choice of area of interest was made based on the need to encompass the major morphological and channel migration patterns in a computationally comfortable spatial domain.

The Padma River, formed by the merge of the Ganges and the Jamuna at Goalundo Ghat, flows southeast for about 120 km before merging with the Meghna River to discharge into the Bay of Bengal. It annually carries millions of tons of sediment from the Himalayan mountain range and has dynamic morphodynamics with strong channel migration, bank erosion, and the regular development of bars. The river discharge is monsoonal, with the peak discharge from July to September, and the river sustains a braided channel with constant channel changes and islands.

The Padma River corridor has dense population, where many settlements exist along both banks. The Padma River sustains important roles in terms of transporting people and goods, irrigation of agricultural lands, fishing, and the supply of potable water. Although it has such importance, the river has high erosion rates, which pose challenges to communities bordering the river, where bank erosion in excess of 500 m can occur in a monsoonal period. This intense erosion has resulted in significant loss of agricultural land, large-scale displacement of households, and destruction of critical infrastructure, including roads, schools, and health facilities [3].

## 4.2 Data Acquisition

### 4.2.1 Landsat Optical Data

Satellite imagery was acquired from two primary remote sensing platforms, divided into two temporally distinct phases based on sensor availability. Phase I (1987–2014) relied exclusively on Landsat optical sensors obtained from the USGS Collection 2 Level-2 surface reflectance archives. Three sensors were employed in succession according to their operational periods: Landsat 5 Thematic Mapper (TM; `LANDSAT/LT05/C02/T1_L2`) for the period prior to 1999, Landsat 7 Enhanced Thematic Mapper Plus (ETM+; `LANDSAT/LE07/C02/T1_L2`) from 1999 onward, and Landsat 8 Operational Land Imager (OLI; `LANDSAT/LC08/C02/T1_L2`) from 2013 onward. All three sensors provided 60 m spatial resolution imagery with a 16-day revisit cycle.

For composite generation across all temporal aggregation levels (yearly, quarterly, and bi-monthly), scene-level cloud filtering was applied using the `CLOUD_COVER` metadata attribute, with a maximum permissible cloud fraction of 50%. The green and shortwave infrared (SWIR1) bands were selected for water index computation, with band designations varying by sensor: `SR_B2` (green) and `SR_B5` (SWIR1) for Landsat 5 and Landsat 7, and `SR_B3` (green) and `SR_B6` (SWIR1) for Landsat 8.

Pixel-level cloud and cloud-shadow removal was applied using the `QA_PIXEL` band via bitwise masking, targeting bit 3 (cloud shadow) and bit 4 (cloud). Collection 2 Level-2 digital numbers were converted to surface reflectance using the standard linear scaling:

```
ρ = 0.0000275 × DN − 0.2        (4.1)
```

For the quarterly composite pipeline, multiple Landsat collections were merged when their operational periods overlapped within a given quarter, ensuring maximum scene availability. For bi-monthly composites (Phase I, 2000–2014), the same sensor selection logic and band-renaming scheme were applied, with six two-month periods per year defined as BM1 (January–February), BM2 (March–April), BM3 (May–June), BM4 (July–August), BM5 (September–October), and BM6 (November–December).

### 4.2.2 Sentinel Optical and SAR Data

Phase II (2015–2025) utilized Sentinel-2 MultiSpectral Instrument (MSI) data as the primary optical source, accessed via the `COPERNICUS/S2_SR_HARMONIZED` collection. Images were filtered using a maximum cloud-cover threshold of 50% based on the `CLOUDY_PIXEL_PERCENTAGE` attribute and were subsequently sorted in ascending order of cloud fraction to prioritize the least contaminated scenes. To prevent computational timeouts during export, the number of Sentinel-2 scenes used per temporal composite was capped at 25 images.

The green band (B3) and the SWIR1 band (B11) were selected for computing the Modified Normalized Difference Water Index (MNDWI). Pixel-level cloud masking was applied using the QA60 band, targeting bit 10 (opaque cloud) and bit 11 (cirrus). Sentinel-2 data were already provided as harmonized surface reflectance and therefore required no additional radiometric scaling.

To fill data gaps arising from persistent cloud cover, specifically during the monsoon season, Sentinel-1 Synthetic Aperture Radar (SAR) data were incorporated as a supplementary fallback when Sentinel-2 imagery was unavailable. Sentinel-1 data were acquired from the `COPERNICUS/S1_GRD` collection in Interferometric Wide (IW) swath mode with VV and VH dual polarization in descending orbit pass. Scenes were sorted by acquisition date in descending order and limited to a maximum of 20 images per temporal composite to manage processing load.

The SAR-based fallback was activated exclusively when no Sentinel-2 scenes passed the cloud filter for a given temporal window. In such cases, a water mask was derived from the VV polarization backscatter using a fixed threshold of −16 dB.

All water masks across both phases were exported as GeoTIFF files in EPSG:4326. An export scale of 60 m was used as the primary spatial resolution to manage file size, with an automated fallback to the bounding-box extent in cases where computational limits were exceeded.

### 4.2.3 Image Availability and Quality Assessment for Time-Series Analysis

*Table 4.1: Summary of image availability and quality across different temporal resolutions.*

| Temporal Resolution | Usable Images | Missing Images | Partial / Low-Quality Images |
|---------------------|---------------|----------------|------------------------------|
| Monthly | 284 | 50 | 122 |
| Bi-monthly | 156 | 16 | 56 |
| Quarterly | 125 | 3 | 24 |
| Yearly | 38 | 0 | 0 |

In order to assure data quality, a first stage of automated filtering was performed using various technical conditions such as presence of clouds or completeness of the imagery. Nonetheless, even with all the filters above applied, there were still some instances where there were partial spatial missing data or regions of missing data.

For this reason, a second manual check was performed in order to detect such instances and improve dataset quality. Those images with partial or missing spatial coverage or degraded were classified under partial or low-quality. As in the case of visual evaluation which is subjective in borderline cases, the numbers reported here are approximate. Figure 4.2 shows two examples of partial or degraded images found in the dataset. The samples shown below represent cases of localized missing data within valid water mask images.

*Figure 4.2: Two representative examples of partially missing.*

Table 4.1 summarizes the availability and quality of images across different temporal resolutions. The monthly dataset provides the highest temporal resolution but also exhibits the greatest data loss, with substantial missing and low-quality observations. The bi-monthly aggregation improves overall completeness but still retains a notable proportion of degraded images. In contrast, the quarterly dataset demonstrates the highest data reliability, with minimal missing observations and relatively few quality issues. Notably, the yearly dataset was found to be fully complete, with no missing or degraded images, making it the most stable representation for long-term temporal analysis.

## 4.3 Data Preprocessing

### 4.3.1 Water Body Detection

We performed water body detection using the Modified Normalized Difference Water Index (MNDWI). This separates water from non-water features such as built-up areas and vegetation [2, 16]. MNDWI is calculated as

```
MNDWI = (Green − SWIR) / (Green + SWIR)        (4.2)
```

where water reflects strongly in the green band and absorbs in the SWIR band, which produces high positive values, while soil, vegetation, and built-up areas yield low or negative values [12, 22].

### 4.3.2 Gap-Filling Process

In this paper, we test a complete suite of gap-filling algorithms for annual water mask images between 1988 to 2025, comprising 38 annual images. To create a real-world missing data scenario, 20% of time series (seven randomly selected years: 2002, 2005, 2010, 2015, 2019, 2020, and 2024) were dropped from the data set and the rest of the data set was used for evaluation and gap filling.

A total of seven gap filling methods have been tested including classical interpolation techniques to deep spatiotemporal learning models:

1. **Mean Composite** and **Median Composite**: This is a simple approach to fill missing yearly mask using a composite of adjacent available images and is a robust albeit naïve approach, utilizing planetary-scale temporal reduction frameworks [32].
2. **Linear Interpolation**: In this approach, linear interpolation between two adjacent available pixels are performed at each point in time, serving as a standard temporal baseline for remote sensing data gaps [33].
3. **Spline Interpolation**: Cubic spline interpolation is performed along the temporal direction to smooth temporal variations and reconstruct missing values in the satellite time series [34].
4. **SDF Interpolation**: A geometric method that converts the binary water mask data into signed distance fields, where temporal interpolation is performed in continuous space, followed by the re-conversion to a binary format [35]. The approach maintains the geometry and topology of water bodies.
5. **Inverse Distance Weighted Temporal Fusion**: The missing water mask data for a particular year are generated through weighted averaging of all the years in temporal terms with weight inversely proportionate to the temporal distance, adapted from classical spatiotemporal IDW models [36].
6. **BiConvLSTM Deep Learning**: A bi-directional Convolutional LSTM network is trained on the spatiotemporal patches of size 128×128 in order to learn temporal dynamics in water distribution [37]. The model uses the approaches of patch based training, masked sequence learning with 20% missing simulation and binary cross entropy objective function.

**Quantitative Evaluation:** Performance was assessed using Intersection over Union (IoU), Dice coefficient, precision, and recall over all held-out years. The results are summarized in Table 4.2.

*Table 4.2: Performance comparison of gap-filling methods.*

| Method | IoU | Dice | Precision | Recall |
|--------|-----|------|-----------|--------|
| Mean Composite | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| Median Composite | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| Linear Interpolation | 0.6548 | 0.7910 | 0.8913 | 0.7193 |
| Spline Interpolation | 0.6733 | 0.8042 | 0.8082 | 0.8012 |
| SDF | 0.7290 | 0.8429 | 0.8741 | 0.8161 |
| Weighted Temporal | 0.6724 | 0.8035 | 0.8306 | 0.7790 |
| **BiConvLSTM** | **0.7366** | **0.8477** | 0.8457 | 0.8511 |

The BiConvLSTM model achieved the best overall performance, with an IoU of 0.7366 and Dice score of 0.8477, demonstrating strong capability in capturing complex spatiotemporal dynamics. Among classical methods, the SDF approach performed the strongest, achieving an IoU of 0.7290 and a Dice score of 0.8429, closely approaching deep learning performance while maintaining structural consistency.

---

# 5. Methodology

## 5.1 Sequence Generation Strategy

To model the temporal evolution of riverbank morphology, the preprocessed binary water masks were transformed into supervised learning sequences using a sliding window approach [38]. Unlike traditional long-sequence forecasting, this study explores multiple short-term temporal dependencies by considering variable sequence lengths adapted to each temporal resolution: L ∈ {4, 5, 6} for yearly data, L ∈ {6, 8, 10} for quarterly data, and L ∈ {6, 9, 12} for bi-monthly data. For each sequence, L consecutive annual observations are used as input, and the immediate next year (t + 1) is selected as the prediction target. Formally, given a temporal sequence {X₁, X₂, ..., X_T}, the input-output pairs are constructed as:

```
Xᵢ = {Xᵢ, Xᵢ₊₁, ..., Xᵢ₊L₋₁},   yᵢ = Xᵢ₊L        (5.1)
```

Stride size is chosen to be 1 year such that overlapping windows will be produced to ensure maximum usage of the temporal data provided. This enables the model to learn short- to mid-term dynamics of the river while still having enough samples to train the model. An approach with multiple sequences is used to find the best temporal context. The model is trained on each of the sequences separately.

```
Seq i:      Xᵢ    Xᵢ₊₁   ···   Xᵢ₊L₋₁  |  Xᵢ₊L     ···
            [--------- Input (L frames) --------]   [- Target (y_i) -]
Seq i+1:    Xᵢ    Xᵢ₊₁   Xᵢ₊₂   ···   Xᵢ₊L    |  Xᵢ₊L₊₁
            [------------ Stride = 1 year ------------]
```

## 5.2 Temporal Data Splitting Strategy

*Table 5.1: Data splitting strategy.*

| Setup | Training | Validation | Testing |
|-------|----------|------------|---------|
| Setup 1 | Up to 2015 (85%) | Up to 2015 (15%) | 2016–2025 |
| Setup 2 | Up to 2020 (85%) | Up to 2020 (15%) | 2021–2025 |

To ensure realistic model evaluation and prevent data leakage, a strict temporal split strategy is adopted. Two experimental setups are considered, as summarized in Table 5.1. The training and validation sets are formed based on chronological ordering without any shuffling, ensuring that all training samples precede validation samples and that both precede the testing period. This temporal separation eliminates look-ahead bias and prevents data leakage during model development and evaluation.

## 5.3 Evaluation Metrics

The performance of the models is evaluated using several pixel-wise segmentation metrics. Predicted probability maps are converted into binary masks using a threshold of 0.5.

- **Intersection over Union (IoU):** IoU measures the overlap between the predicted and ground-truth masks and is defined as

```
IoU = (Intersection + ε) / (Union + ε)        (5.2)
```

where ε is a small constant added to avoid division by zero.

- **Dice Coefficient:** The Dice coefficient quantifies the similarity between the predicted and ground-truth masks:

```
Dice = 2 TP / (2 TP + FP + FN)        (5.3)
```

- **Precision:** Precision measures the proportion of correctly predicted positive pixels among all predicted positive pixels:

```
Precision = TP / (TP + FP)        (5.4)
```

- **Recall:** Recall measures the proportion of correctly predicted positive pixels among all actual positive pixels:

```
Recall = TP / (TP + FN)        (5.5)
```

In addition to the pixel-wise metrics, the difference between the predicted and actual river area is computed as

```
ΔA = A_pred − A_true        (5.6)
```

where A_pred and A_true denote the predicted and ground-truth water-covered areas (km²), respectively.

## 5.4 Model Architectures

This study shows a comparative deep learning framework to model riverbank evolution using five architectures: (i) a standalone ConvLSTM model, (ii) U-Net with ConvLSTM, (iii) Attention U-Net with ConvLSTM, (iv) a Swin Spatio-Temporal Transformer, and (v) a Vision Transformer (ViT)-based spatio-temporal model. These models are selected to analyze different facets of spatiotemporal learning, spanning recurrent convolutional layers, hybrid convolution-attention networks, and pure transformer-based architectures. All models are trained and evaluated under identical preprocessing, sequence generation, and temporal splitting strategies to ensure a fair and consistent comparison.

## 5.5 ConvLSTM Network Architecture

The ConvLSTM model explicitly captures spatiotemporal dependencies by integrating convolutional operations within recurrent units, enabling direct modeling of temporal evolution [9].

*Figure 5.1: ConvLSTM Architecture.*

```
Input Sequence (T frames)
        |
Conv2D(32) + BN + MaxPool + Dropout
        |
Conv2D(64) + BN + MaxPool + Dropout
        |
ConvLSTM2D(128) + BN  (Return Sequences = True)
        |
ConvLSTM2D(64) + BN   (Return Sequences = False)
        |
Upsampling + Conv2D(64) + BN + Dropout
        |
Upsampling + Conv2D(32) + BN + Dropout
        |
Conv2D(16) + ReLU
        |
Conv2D(1) + Sigmoid
        |
Predicted Mask
```

## 5.6 U-Net with ConvLSTM

The model U-Net + ConvLSTM relies on the Time Distributed encoder for handling temporal sequences, utilizes LSTM at the bottleneck layer to model temporal relationships, and makes predictions through the decoder with skip connections. The U-Net model focuses primarily on spatial feature extraction using an encoder–decoder structure with temporal modeling at the bottleneck via LSTM [29, 30].

*Figure 5.2: U-Net + ConvLSTM Architecture.*

```
Input Sequence (T frames)
        |
Enc1: 2x(Conv2D(32)+BN) + Pool + Dropout
        |
Enc2: 2x(Conv2D(64)+BN) + Pool + Dropout
        |
Enc3: 2x(Conv2D(128)+BN) + Pool + Dropout
        |
Bottleneck: Conv2D(256) + BN
            2x ConvLSTM2D (256 -> 128 Filters) + BN
        |
Dec3: Up + Cat(Enc3) + 2x(Conv2D(128)+BN) + Drop     [Lambda (T-1)]
        |
Dec2: Up + Cat(Enc2) + 2x(Conv2D(64)+BN) + Drop      [Lambda (T-1)]
        |
Dec1: Up + Cat(Enc1) + 2x(Conv2D(32)+BN)             [Lambda (T-1)]
        |
Conv2D(16) + ReLU -> Conv2D(1) + Sigmoid
        |
Predicted Mask
```

## 5.7 Attention U-Net with ConvLSTM

The Attention U-Net further enhances spatial representation by incorporating attention mechanisms into skip connections, while its integration with ConvLSTM introduces temporal awareness [23].

*Figure 5.3: Attention U-Net + ConvLSTM Architecture.*

```
Input Sequence (T frames)
        |
Enc1: Conv2D(32) + BN + Conv2D(32) + Pool + Drop
        |
Enc2: Conv2D(64) + BN + Conv2D(64) + Pool + Drop
        |
Enc3: Conv2D(128) + BN + Pool + Drop
        |
Bottleneck: TimeDistributed Conv2D(256)
            2x ConvLSTM2D (256 -> 128 Filters) + BN
        |
  [Attention Gate 3] [Attention Gate 2] [Attention Gate 1]
        |
Dec3: Up(2) + Cat(Att3) + Conv2D(128) + BN + Drop
        |
Dec2: Up(2) + Cat(Att2) + Conv2D(64) + BN + Drop
        |
Dec1: Up(2) + Cat(Att1) + Conv2D(32) + BN
        |
Conv2D(16) + ReLU -> Conv2D(1) + Sigmoid
        |
Predicted Mask
```

### 5.7.1 Swin Transformer

The Swin Spatio-Temporal Transformer employs hierarchical window-based self-attention for multi-scale spatial feature extraction, followed by a dedicated temporal fusion module [39].

*Figure 5.4: Swin Spatio-Temporal Transformer architecture with a shared encoder, temporal fusion module, and U-Net style decoder.*

```
Input (B, T, 256, 256, 1)
        |
Patch Embedding 4x4 -> (B, 64, 64, 64)
        |
Swin Stage 1: 2x Window Attention (h=2) -> skip1 (64, 64, 64)
        |
Patch Merging /2 + Swin Stage 2: 2x Window Attention (h=4) -> skip2 (32, 32, 128)
        |
Patch Merging /2 + Swin Stage 3: 2x Window Attention (h=8) -> (16, 16, 256)
        |
Stack T feature maps -> (B, T, 16, 16, 256)
        |
Temporal Conv1D (k=3, GELU)
        |
2x Cross-Temporal Attention (per-pixel across T, h=4)
        |
Temporal Aggregation (Softmax-weighted) -> (B, 16, 16, 256)
        |
x2 + Concat(skip2) + Conv(128)x2
        |
x2 + Concat(skip1) + Conv(64)x2
        |
x2 + Conv(32) -> (128, 128, 32)
        |
x2 + Conv(16) -> (256, 256, 16)
        |
Conv2D(1, 1x1) + Sigmoid
        |
Predicted Mask (B, 256, 256, 1)
```

### 5.7.2 Vision Transformer (ViT)-Based Spatio-Temporal Model

The proposed ViT-based spatio-temporal model combines per-frame spatial tokenization with a dedicated temporal reasoning module, enabling joint modelling of spatial structure and temporal dynamics from image sequences [40, 41].

*Figure 5.5: ViT-based spatio-temporal architecture with per-frame spatial tokenization, temporal transformer fusion, and convolutional decoder.*

```
Input (B, T, 256, 256, 1)
        |
Frame Slice (B, 256, 256, 1)        [Spatial Encoder xT]
        |
Patch Embedding 16x16 -> (B, 256, 128)
        |
Spatial Transformer Block 1
        |
Spatial Transformer Block 2
        |
Global Average Pooling -> (B, 128)
        |
Stack Frame Tokens -> (B, T, 128)
        |
Temporal Positional Embedding
        |
Temporal Transformer Block 1
        |
Temporal Transformer Block 2
        |
Last Token / Sequence Summary -> (B, 128)
        |
Dense Projection -> (16, 16, 128)
        |
x2 + ConvT 128 -> x2 + ConvT 64 -> x2 + ConvT 32 -> x2 + ConvT 16
        |
Refine Convolution Block
        |
Conv2D(1, 1x1) + Sigmoid
        |
Predicted Mask (B, 256, 256, 1)
```

## 5.8 Loss Functions

### 5.8.1 Dice Loss

Dice loss directly optimizes the Dice coefficient, focusing on the overlap between the predicted mask and ground truth:

```
L_Dice = 1 − (2|A ∩ B| + ε) / (|A| + |B| + ε)        (5.7)
```

where ε is a smoothing term to prevent division by zero. Dice loss is particularly effective for handling class imbalance in segmentation tasks where the target feature (e.g., water bodies) occupies a small fraction of the total image area.

### 5.8.2 Binary Cross-Entropy Loss

Binary Cross-Entropy (BCE) loss measures the performance of the classification model whose output is a probability value between 0 and 1:

```
L_BCE = −(1/N) Σᵢ₌₁ᴺ [yᵢ log(pᵢ) + (1 − yᵢ) log(1 − pᵢ)]        (5.8)
```

where yᵢ is the true binary label and pᵢ is the predicted probability for pixel i. BCE evaluates pixel-wise accuracy independently, providing stable gradients during the early stages of training.

### 5.8.3 Combined Loss

In practice, the network is optimized using a joint loss function that combines both pixel-level and region-level objectives:

```
L_Total = L_BCE + L_Dice        (5.9)
```

We used this combined loss function because it leverages the complementary strengths of each individual component—balancing independent pixel-wise accuracy from BCE with region-based boundary precision from Dice loss—to produce more stable and reliable segmentation performance across diverse riverbank geometries.

## 5.9 Training Configuration

All models were trained using identical preprocessing, sequence generation, and strict temporal split strategies to ensure a fair comparison across architectures. The input raster masks were resized to 256×256, and binary segmentation masks were generated after connected-component cleaning by retaining the three largest components. Multi-temporal input sequences were evaluated with a one-step prediction horizon and stride of 1. Two strict temporal evaluation setups were used: Setup 1 trained on samples up to 2015 and tested on 2016–2025, while Setup 2 trained up to 2020 and tested on 2021–2025. Validation samples were created from the final 15% of the temporally ordered training set to prevent temporal leakage.

For the ConvLSTM, U-Net + ConvLSTM, and Attention U-Net + ConvLSTM models, the same training configuration was applied. These models were optimized using the Adam optimizer with a fixed learning rate of η = 1×10⁻⁴ and gradient clipping using clipnorm = 1.0. A batch size of 4 sequences was used, and the maximum training duration was set to 200 epochs. The loss function was a hybrid Binary Cross-Entropy (BCE) and Dice loss formulation, while Dice coefficient and Intersection over Union (IoU) were used as evaluation metrics. Early stopping monitored validation loss with a patience of 20 epochs and restored the best weights. Learning rate reduction was triggered using ReduceLROnPlateau with patience of 7 epochs, reduction factor of 0.5, and a minimum learning rate of 1×10⁻⁷. Model checkpointing was used to save the best-performing model based on validation loss.

The Swin Spatio-Temporal Transformer employed a slightly different configuration due to its higher computational complexity. Training used the Adam optimizer with the same learning rate of η = 1×10⁻⁴ and clipnorm = 1.0, but with a reduced batch size of 2 sequences to accommodate higher GPU memory requirements. The model was trained for up to 200 epochs with the same callback strategy. Spatial encoding used patch size 4, embedding dimension 64, window size 8, hierarchical Swin depths of [2, 2, 2], attention heads of [2, 4, 8], MLP ratio of 2.0, and spatial dropout of 0.1. Temporal fusion was performed using 2 temporal transformer layers with 4 attention heads, MLP dimension 256, and dropout of 0.1. Decoder filters were configured as [128, 64, 32, 16].

For the Vision Transformer (ViT)-based spatio-temporal model, training also used Adam optimization with learning rate η = 1×10⁻⁴, clipnorm = 1.0, batch size of 2, and a maximum of 200 epochs with the same early stopping, learning rate scheduling, and checkpointing strategy. The ViT encoder used a patch size of 16 and embedding dimension of 128. Spatial feature extraction employed 2 transformer layers with 4 multi-head self-attention heads, while temporal aggregation used 2 temporal transformer layers with 4 heads. The MLP hidden dimension was set to 256 with transformer dropout of 0.1.

All implementations were developed using fixed random seeds for reproducibility, and identical data splits for consistent comparative evaluation across all five architectures.

## 5.10 Prediction Process

The prediction process is based on learning spatiotemporal patterns from a sequence of historical binary raster images representing land and water distribution over time. The model receives a fixed-length sequence of past observations and learns both spatial structures (such as riverbank shape and geometry) and temporal dependencies (how these structures evolve across successive time steps). During inference, the trained system uses the most recent observed sequence to generate the next future spatial map, and this process can be repeated iteratively to produce multi-step future forecasts. The model effectively captures nonlinear changes in river morphology by combining spatial feature extraction with temporal memory, enabling it to simulate progressive erosion and accretion dynamics over time. The post-processing process after map generation involves the comparison of sequential maps on a pixel-by-pixel basis, which helps in recognizing transitions between land and water. These transitions help in understanding the dynamic nature of the river system by determining the locations where the transition from land to water occurs, indicating erosion, and from water to land, indicating deposition or accretion. This method involves the computation of normalized frequencies for all the transitions over the entire forecast period to generate risk zones.

## 5.11 Risk Zone Mapping

The process of predicting future scenarios of the rivers' banks for the time frame of 2026–2040 through the use of the trained deep learning networks is followed by the post-processing stage that identifies erosion and accretion risk zones based on spatiotemporal predictions. In this research, risk is estimated at the pixel level using temporal changes between land and water binary conditions, where 1 is the land and 0 stands for water. The erosion risk zone can be identified by the pixel changing from land to water, namely Lₜ = 1 and Lₜ₊₁ = 0, while the accretion risk zone implies the change in the reverse direction, which is Lₜ = 0 and Lₜ₊₁ = 1. The frequency of the changes in the geomorphology is calculated based on the full 15-year time interval in order to identify the risk probability. Erosion and accretion frequencies are defined by the formulas:

```
Erosion Frequency   = (1/T) Σₜ₌₁ᵀ I(Lₜ = 1 ∧ Lₜ₊₁ = 0)        (5.10)
Accretion Frequency = (1/T) Σₜ₌₁ᵀ I(Lₜ = 0 ∧ Lₜ₊₁ = 1)        (5.11)
```

where T represents the total number of predicted years (T = 15). These frequency maps capture the temporal persistence of erosion and deposition processes, providing a probabilistic interpretation of long-term river migration behavior.

Alongside the transition probability approach, a hydrodynamic stability index is computed to capture uncertainty in pixel-wise classification over time. Let F_w denote the fraction of time a pixel is predicted as water across the forecast horizon, defined as F_w = (1/T) Σₜ₌₁ᵀ (1 − Lₜ). The instability of each pixel is then quantified using a symmetric formulation centered at F_w = 0.5, given by I = 1 − |2F_w − 1|, where values closer to 1 indicate highly unstable regions with frequent land–water oscillations, and values near 0 represent stable zones dominated by either land or water.

To estimate net geomorphological change across key tracking milestones, the predicted maps at selected temporal intervals are compared with the last observed baseline, allowing identification of cumulative stable river structures, erosion, and accretion zones. Pixels that remain land across the evaluation are marked as stable, those transitioning from land in the observed baseline state to water in the prediction are marked as cumulative erosion zones, while water-to-land transitions represent accretion zones. These changes are further converted into physical area estimates by multiplying the number of affected pixels with the spatial resolution-based pixel area, i.e., A_erosion = N_erosion × A_pixel and A_accretion = N_accretion × A_pixel.

Lastly, the composite risk map is generated by combining erosion frequency, accretion frequency, and instability layers in an interactive geospatial map through Folium. The generated map includes the combination of probabilities, risk zones classification, and dots of risk tracking on the background of the satellite image, emphasizing the areas that are both dynamic and risky at the same time. It is the approach that converts spatiotemporal forecasts based on deep learning into risk maps for geomorphological risk assessment.

---

# 6. Spatiotemporal Statistical Analysis of Padma River Dynamics

## 6.1 Results

### 6.1.1 Areal Dynamics and Seasonal Variability

The Padma River showed considerable inter-annual variability in water-covered area over the 38-year observation period (1988–2025). Annual water-covered area ranged from 540.4 km² (1995, minimum) to 1,229.3 km² (1999, maximum), with a long-term mean of 664.5 ± 112.8 km² (1σ). The exceptionally large expansion in 1999 corresponds to the major monsoon flood event, while the minimum extent in 1995 reflects a comparatively dry hydrological year.

As for the presence of any trends, the Mann–Kendall test did not show any statistically significant trend of monotonic type (τ = −0.149, p = 0.187), and the Theil–Sen estimator showed only a weak negative slope of −1.41 km² yr⁻¹ (Fig. 6.1). All in all, these findings imply that although strong inter-annual fluctuations occur, the long-term mean river extent has remained broadly stable, with no evidence of sustained expansion or contraction over the study period.

*Figure 6.1: Long-term annual water area.*

The seasonal variability, assessed through coefficient of variation (CV) of bi-monthly water area, also showed a significant level of fluctuation between years. The annual CV ranged from 8.2% (2018, most stable year) to 64.1% (2007, most variable year), with a study-period mean of 34.3%. Years with high CV values, particularly 2007, 2011, and 2020, were linked to high magnitude of floods due to monsoons, during which bi-monthly water extent exceeded 2,700 km² during peak inundation. In contrast, low-CV years indicate relatively stable seasonal water extent and weaker flood-pulse variability.

The quarterly heat-map reveals a similar hydrological pattern across all years: Q4 (October–December) consistently exhibits the largest water extent, while the second quarter (April–June) generally records the minimum extent (Fig. 6.3). This seasonal structure reflects the dominant influence of the South Asian monsoon system on Padma River hydrodynamics.

*Figure 6.2: Seasonal water area dynamics of the Padma River (1988–2025). Left: bi-monthly water area profiles for all years, coloured by year. Right: annual coefficient of variation (CV) of bi-monthly water area, representing inter-annual differences in flood-pulse intensity and hydrological variability.*

*Figure 6.3: Quarterly heat-map of water area.*

### 6.1.2 Spatial Change Analysis: Erosion and Accretion

Annual erosion and accretion rates demonstrated very high inter-annual variability, reflecting the highly dynamic planform behaviour of the Padma River channel (Fig. 6.4). The largest single-year erosion event occurred during 1998–1999, when 649.1 km² of land was converted to water, resulting in a net channel expansion of 567.7 km²—the most extreme widening event recorded during the study period. This major adjustment corresponds to the catastrophic 1998 flood and subsequent geomorphic reworking.

Conversely, the following year (1999–2000) recorded the largest net accretion (−503.5 km²), indicating substantial channel contraction immediately after the flood peak. This rapid reversal highlights the strong recovery phase following extreme hydrological disturbance.

*Figure 6.4: Annual and decadal erosion and accretion of the channel.*

Through the complete period under review, the cumulative gross erosion was 5,170.2 km², while cumulative gross accretion totalled 5,323.3 km², producing an overall net accretion of 153.0 km². This implies that, despite the many instances of considerable widening and meandering of the river, the general trend of the Padma River was slightly towards channel narrowing.

Decadal analysis reveals four distinct morphodynamic phases (Table 6.1). During 1988–1998, the river experienced moderate net accretion (−66.7 km²), indicating near-balanced erosion–accretion dynamics associated with lateral channel migration. The 1998–2008 decade shifted toward slight net erosion (+26.3 km²), largely driven by the major 1998–1999 flood event and its post-flood adjustment.

The strongest net accretion occurred during 2008–2018 (−113.0 km²), suggesting significant channel contraction and partial stabilization. The most recent period (2018–2025) approached geomorphic equilibrium, with a negligible net change of only +0.3 km², indicating that recent channel behaviour has become comparatively stable.

### 6.1.3 Morphological Metrics and Channel Migration

Channel centreline migration rates exhibited strong temporal variability, reflecting alternating periods of instability and relative stabilization (Fig. 6.5). The mean annual migration rate was 255.4 m yr⁻¹, with values ranging from 134.2 m yr⁻¹ (2001–2002 and 2022–2023) to 1,485.5 m yr⁻¹ during 1998–1999.

The extreme migration observed in 1998–1999 coincided with the catastrophic flood event and represents the most intense lateral adjustment recorded during the study period. In contrast, the low migration rates observed during the early 2000s and recent years indicate phases of temporary channel stabilization.

*Table 6.1: Decadal erosion, accretion, and net channel change.*

| Period | Erosion (km²) | Accretion (km²) | Net change (km²) |
|--------|---------------|-----------------|------------------|
| 1988–1998 | 367.6 | 434.2 | −66.7 |
| 1998–2008 | 382.3 | 355.9 | +26.3 |
| 2008–2018 | 262.8 | 375.8 | −113.0 |
| 2018–2025 | 264.6 | 264.3 | +0.3 |
| **Total** | **5170.2** | **5323.3** | **−153.0** |

The median migration rate (216.3 m yr⁻¹) was very lower than the mean, which indicates that a small number of extreme flood years strongly influenced long-term averages. The 1990s recorded the highest average migration rates, that confirms that this decade represented the period of maximum geomorphic instability.

Cumulative centreline displacement over the full study period exceeded 9 km, demonstrating sustained long-term lateral adjustment and continuous reworking of the active river corridor.

*Figure 6.5: Annual lateral migration rate of the centreline.*

---

# 7. Experimental Results and Analysis

## 7.1 Yearly Data (Long-Term Analysis)

To evaluate long-term satellite image prediction performance, experiments were conducted using yearly composite data under two forecasting setups: (i) Setup 1: 10-year prediction and (ii) Setup 2: 5-year prediction. Five deep learning architectures were evaluated: ConvLSTM, U-Net+LSTM, Attention U-Net+ConvLSTM, Vision Transformer (ViT), and Swin Transformer. Model performance was assessed using Intersection over Union (IoU), Dice coefficient, Precision, Recall, and Absolute Area Difference.

For each model, multiple sequence lengths were tested, and the best-performing configuration was selected based on mean IoU. A consistent result presentation structure is maintained for all models, followed by a detailed year-wise analysis of the best-performing model.

### ConvLSTM Results

Table 7.1 presents the best performance of the ConvLSTM model for both forecasting setups.

*Table 7.1: ConvLSTM Performance on Yearly Data.*

| Setup | IoU | Dice | Precision | Recall | Area Diff (km²) |
|-------|-----|------|-----------|--------|-----------------|
| Setup 1 (10-year) | 0.6218 | 0.7665 | 0.7212 | 0.8207 | 76.69 |
| Setup 2 (5-year) | 0.6278 | 0.7711 | 0.7139 | 0.8432 | 103.26 |

Moderate segmentation accuracy was shown by the ConvLSTM model in both experimental setups. The better IoU and Dice scores were obtained in the experiment with shorter prediction horizon (Setup 2), which implies a higher overlap of the predicted region boundaries and ground truth. Nevertheless, the larger value of the absolute area difference shows the worse spatial precision in comparison with other models.

### U-Net+LSTM Results

Table 7.2 summarizes the best results obtained using the U-Net+LSTM model.

*Table 7.2: U-Net+LSTM Performance on Yearly Data.*

| Setup | IoU | Dice | Precision | Recall | Area Diff (km²) |
|-------|-----|------|-----------|--------|-----------------|
| Setup 1 (10-year) | 0.6964 | 0.8208 | 0.7876 | 0.8595 | 50.25 |
| Setup 2 (5-year) | 0.6986 | 0.8223 | 0.7905 | 0.8592 | 48.61 |

Good and stable performance was reached by the U-Net+LSTM model in both prediction horizons. Comparing the results of this architecture with those of ConvLSTM, it is possible to state that there is a significant improvement in IoU, Dice and precision and relative low values of the area difference. Thus, the encoder-decoder architecture of U-Net helps to extract spatial features and improve the accuracy of boundaries' prediction.

### Attention U-Net+ConvLSTM Results

The best results of the Attention U-Net+ConvLSTM model are presented in Table 7.3.

*Table 7.3: Attention U-Net+ConvLSTM Performance on Yearly Data.*

| Setup | IoU | Dice | Precision | Recall | Area Diff (km²) |
|-------|-----|------|-----------|--------|-----------------|
| Setup 1 (10-year) | 0.7005 | 0.8236 | 0.7807 | 0.8747 | 66.54 |
| Setup 2 (5-year) | 0.6963 | 0.8205 | 0.7934 | 0.8523 | 41.05 |

It has the largest values of IoU and Dice score in Setup 1 in comparison with other proposed architectures, which means that the model has good forecasting ability. The attention mechanism provides the possibility to select important features, and ConvLSTM takes into account temporal dependencies. Although Setup 1 showed the best values of overlap metrics, Setup 2 provided smaller values of the area difference.

### Vision Transformer Results

Table 7.4 presents the best yearly prediction performance of the Vision Transformer (ViT) model.

*Table 7.4: Vision Transformer Performance on Yearly Data.*

| Setup | IoU | Dice | Precision | Recall | Area Diff (km²) |
|-------|-----|------|-----------|--------|-----------------|
| Setup 1 (10-year) | 0.3613 | 0.5302 | 0.4352 | 0.6806 | 316.06 |
| Setup 2 (5-year) | 0.3831 | 0.5526 | 0.5197 | 0.5907 | 78.69 |

The Vision Transformer produced the lowest performance among all evaluated models. Both setups achieved their best results using a sequence length of 4. In Setup 1, the model showed low IoU and Dice scores along with a very high area difference, indicating significant overestimation of river extent. Although Setup 2 showed slight improvement and a reduced area difference, the overall segmentation quality remained comparatively weak. This suggests that the ViT model struggled to effectively capture fine spatial structures in yearly prediction.

### Swin Transformer Results

The best performance of the Swin Transformer model is shown in Table 7.5.

*Table 7.5: Swin Transformer Performance on Yearly Data.*

| Setup | IoU | Dice | Precision | Recall | Area Diff (km²) |
|-------|-----|------|-----------|--------|-----------------|
| Setup 1 (10-year) | 0.6080 | 0.7560 | 0.7364 | 0.7780 | 31.64 |
| Setup 2 (5-year) | 0.5912 | 0.7431 | 0.6849 | 0.8122 | 96.23 |

The Swin Transformer achieved better performance than ViT and ConvLSTM, particularly in Setup 1 where it obtained a relatively high IoU and the lowest area difference among all models. This indicates strong spatial calibration despite slightly lower overlap metrics than U-Net-based approaches. In Setup 2, however, the model showed reduced IoU and a higher area difference, suggesting less stable performance for shorter-term prediction. The hierarchical attention structure contributes to improved spatial representation, but temporal consistency remains comparatively weaker than hybrid CNN-LSTM models.

### Best Model Selection

Of all the tested architectures, the Attention U-Net+ConvLSTM architecture was considered the best performing one for river prediction on an annual basis. The IoU and Dice coefficient values were higher (IoU = 0.7005 and Dice coefficient = 0.8236) in the 10-year prediction task despite the high performance observed in the 5-year task. While U-Net+LSTM performed consistently, the attention mechanism was better in terms of learning and segmentation.

To further analyze the robustness of the selected model, detailed year-wise prediction results are presented for both the 10-year and 5-year forecasting setups.

*Table 7.6: Detailed Year-wise Prediction Results (10-Year Prediction: 2016–2025).*

| Year | IoU | Dice | Precision | Recall | Area Diff (km²) |
|------|-----|------|-----------|--------|-----------------|
| 2016 | 0.7357 | 0.8477 | 0.8655 | 0.8307 | -24.96 |
| 2017 | 0.6975 | 0.8218 | 0.7396 | 0.9246 | 130.72 |
| 2018 | 0.6742 | 0.8054 | 0.7273 | 0.9023 | 122.18 |
| 2019 | 0.7051 | 0.8270 | 0.7858 | 0.8729 | 58.46 |
| 2020 | 0.7055 | 0.8273 | 0.7924 | 0.8654 | 51.24 |
| 2021 | 0.6569 | 0.7929 | 0.7706 | 0.8165 | 34.81 |
| 2022 | 0.7359 | 0.8479 | 0.8407 | 0.8552 | 10.84 |
| 2023 | 0.7081 | 0.8291 | 0.7618 | 0.9095 | 107.73 |
| 2024 | 0.7046 | 0.8267 | 0.7806 | 0.8786 | 70.94 |
| 2025 | 0.6814 | 0.8105 | 0.7430 | 0.8915 | 103.46 |

The 10-year prediction results show relatively stable performance across the full forecasting horizon. Higher IoU values were observed in 2016 and 2022, while slightly lower performance occurred in 2021 and 2025. Positive area differences in several years indicate moderate overestimation of river extent, whereas negative values in 2016 suggest slight underestimation. Despite these fluctuations, the model maintained strong Dice and recall values throughout the sequence, confirming reliable long-term prediction capability.

As for the 5-year prediction setup, it showed better spatial calibration than the 10-year setup, as the area differences were lower. Best results were shown in 2022.

*Table 7.7: Detailed Year-wise Prediction Results (5-Year Prediction: 2021–2025).*

| Year | IoU | Dice | Precision | Recall | Area Diff (km²) |
|------|-----|------|-----------|--------|-----------------|
| 2021 | 0.6384 | 0.7793 | 0.7797 | 0.7788 | -0.66 |
| 2022 | 0.7270 | 0.8419 | 0.8546 | 0.8296 | -18.39 |
| 2023 | 0.7143 | 0.8333 | 0.7745 | 0.9018 | 91.31 |
| 2024 | 0.7000 | 0.8235 | 0.7950 | 0.8542 | 42.04 |
| 2025 | 0.7017 | 0.8247 | 0.7631 | 0.8972 | 90.98 |

with maximum IoU and Dice and a negative area difference. While 2023 and 2025 showed larger positive area differences, the overall results remained stable and consistent. This confirms that shorter prediction horizons reduce temporal uncertainty and improve spatial precision.

## 7.2 Quarterly Data (Short-Term Analysis)

To evaluate short-term satellite image prediction performance, experiments were conducted using quarterly composite data under two forecasting setups: (i) Setup 1: long-range quarterly prediction (test from 2015-Q1 onward) and (ii) Setup 2: medium-range quarterly prediction (test from 2020-Q1 onward). Five deep learning architectures were evaluated: ConvLSTM, U-Net+LSTM, Attention U-Net+ConvLSTM, Vision Transformer (ViViT), and Swin Transformer. Model performance was assessed using Intersection over Union (IoU), Dice coefficient, Precision, Recall, and Absolute Area Difference.

For each model, multiple sequence lengths were tested, and the best-performing configuration was selected based on mean IoU. A consistent result presentation structure is maintained for all models, followed by a detailed quarter-wise analysis of the best-performing model.

### ConvLSTM Results

Table 7.8 presents the best performance of the ConvLSTM model for both quarterly forecasting setups.

*Table 7.8: ConvLSTM Performance on Quarterly Data (Best Sequence per Setup).*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.6114 | 0.7572 | 0.7433 | 0.7918 | 166.66 |
| Setup 2 (medium-range) | 0.6134 | 0.7586 | 0.7115 | 0.8348 | 159.91 |

The ConvLSTM model achieved the lowest performance among the hybrid CNN-LSTM architectures. Both setups produced nearly identical IoU values, indicating limited adaptability across different quarterly forecasting horizons. Setup 2 showed higher recall but lower precision, suggesting a tendency to over-predict river extent. The relatively high absolute area differences further confirm reduced spatial accuracy. Compared to yearly prediction, the quarterly setting introduces stronger temporal fluctuations, which ConvLSTM alone struggles to model effectively.

### U-Net+LSTM Results

Table 7.9 summarizes the best results obtained using the U-Net+LSTM model.

*Table 7.9: U-Net+LSTM Performance on Quarterly Data (Best Sequence per Setup).*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.6908 | 0.8095 | 0.8162 | 0.8318 | 188.03 |
| Setup 2 (medium-range) | 0.7008 | 0.8191 | 0.8257 | 0.8431 | 163.76 |

The U-Net+LSTM model demonstrated substantial improvement over ConvLSTM in both setups. IoU increased by approximately 0.08–0.09, reflecting the effectiveness of combining strong spatial feature extraction with temporal modeling. The precision-recall balance remained stable, indicating consistent segmentation quality. Setup 2 achieved the highest IoU for this model, while Setup 1 showed strong generalization over a longer temporal range. These results confirm that the U-Net encoder-decoder structure significantly improves quarterly boundary prediction.

### Attention U-Net+ConvLSTM Results

The best results of the Attention U-Net+ConvLSTM model are presented in Table 7.10.

*Table 7.10: Attention U-Net+ConvLSTM Performance on Quarterly Data (Best Sequence per Setup).*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.6956 | 0.8139 | 0.8095 | 0.8443 | 184.80 |
| Setup 2 (medium-range) | 0.6972 | 0.8170 | 0.8054 | 0.8511 | 137.98 |

The Attention U-Net+ConvLSTM achieved the highest IoU in Setup 1 and the lowest absolute area difference in Setup 2, indicating strong performance across both forecasting horizons. Specifically, the attention mechanism enhances feature selection by highlighting important river areas while the ConvLSTM maintains temporal smoothness. Although Setup 2 resulted in a slightly smaller IoU compared to U-Net+LSTM, it provided the highest spatial consistency with the smallest area difference among all approaches.

### Vision Transformer (ViViT) Results

Table 7.11 presents the best quarterly prediction performance of the Vision Transformer (ViViT) model.

*Table 7.11: Vision Transformer (ViViT) Performance on Quarterly Data (Sequence Length = 6).*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.3648 | 0.5316 | 0.4788 | 0.6347 | 340.08 |
| Setup 2 (medium-range) | 0.3822 | 0.5496 | 0.4606 | 0.7281 | 447.09 |

The lowest performance results were obtained using the Vision Transformer approach in the quarterly setting. Poor performance metrics can be seen both in Setup 1 and in Setup 2 with regards to the IoU and Dice metrics. While the former one shows poor values, the latter one is slightly higher in Setup 2, while precision is poor but recall is relatively high. This tendency reflects in high area differences, especially in Setup 2. It appears that the approach has shown consistent optimization process but failed to identify spatial and temporal features.

### Swin Transformer Results

The best performance of the Swin Transformer model is shown in Table 7.12.

*Table 7.12: Swin Transformer Performance on Quarterly Data (Best Sequence per Setup).*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.5865 | 0.7370 | 0.7012 | 0.7981 | 190.79 |
| Setup 2 (medium-range) | 0.5882 | 0.7392 | 0.7004 | 0.8024 | 160.45 |

The Swin Transformer achieved better performance than ViViT but remained weaker than U-Net-based architectures. In both cases, IoU values were close to 0.59, which implies poor predictive ability when considering quarterly forecasting periods. The high recall rate and low precision imply an inclination towards overestimation of river extents in particular Setup 2. Though the hierarchical self-attention mechanism enhances the overall spatial modeling, it demonstrated a poor temporal behavior and was inferior to hybrid CNN-LSTM models regarding segmentation.

### Best Model Selection

Among all architectures analyzed, the Attention U-Net+ConvLSTM was identified as the best performing model for quarterly predictions. It obtained the highest value of IoU in Setup 1 (more difficult setting) – 0.6956 – and the smallest value of absolute area difference in Setup 2 (137.98 km²). While the U-Net+LSTM architecture had a higher value of IoU in Setup 2 (0.7008), the attention-based architecture demonstrates better results concerning spatial calibration and prediction accuracy in both setups.

The superior performance of the Attention U-Net+ConvLSTM suggests the usefulness of attention-guided feature selection for quarterly prediction because of seasonal dynamics and temporal variations within quarters.

### Detailed Quarterly Prediction Results

Tables 7.13 and 7.14 present the detailed quarter-wise prediction results for the selected Attention U-Net+ConvLSTM model.

The results obtained for Setup 1 are characterized by clear seasonal variations between different quarters. High values of IoU were noticed in Q2, while Q3 shows the worst performance in a number of years. Extreme values of IoU like 0.9298 (2025-Q2) suggest that there is great spatial agreement within good observation conditions, while lower values such as 0.4833 (2021-Q3) are related to the uncertainty caused by seasonal interference. Nevertheless, even though there is considerable variation, the average performance is quite stable with almost no area difference, which suggests high spatial calibration.

In contrast, the results of Setup 2 are characterized by somewhat stable performance due to the shorter forecasting horizon. The maximum IoU value is observed in 2025-Q2, while lower performance is noticed once again in Q3 quarters. Despite this, in comparison with Setup 1, there is an improvement in spatial agreement due to the decrease in forecasting horizon.

*Table 7.13: Detailed Quarterly Prediction Results — Setup 1 (Attention U-Net+ConvLSTM, SeqLen=8).*

| Quarter | IoU | Dice | Precision | Recall | ∆Area (km²) |
|---------|-----|------|-----------|--------|-------------|
| 2017-Q2 | 0.9156 | 0.9559 | 0.9524 | 0.9595 | 5.31 |
| 2017-Q3 | 0.7556 | 0.8608 | 0.9543 | 0.7840 | -171.18 |
| 2017-Q4 | 0.7778 | 0.8750 | 0.9127 | 0.8404 | -73.25 |
| 2018-Q1 | 0.8315 | 0.9080 | 0.8833 | 0.9341 | 41.75 |
| 2018-Q2 | 0.9150 | 0.9556 | 0.9304 | 0.9823 | 38.33 |
| 2018-Q3 | 0.7056 | 0.8274 | 0.8341 | 0.8208 | -12.15 |
| 2018-Q4 | 0.5594 | 0.7175 | 0.6733 | 0.7678 | 94.13 |
| 2019-Q1 | 0.8782 | 0.9351 | 0.8851 | 0.9911 | 71.73 |
| 2019-Q2 | 0.6949 | 0.8200 | 0.7197 | 0.9528 | 197.74 |
| 2019-Q3 | 0.5647 | 0.7218 | 0.9123 | 0.5971 | -447.49 |
| 2019-Q4 | 0.6155 | 0.7620 | 0.6746 | 0.8754 | 241.01 |
| 2020-Q1 | 0.7440 | 0.8532 | 0.7610 | 0.9708 | 157.89 |
| 2020-Q2 | 0.9068 | 0.9511 | 0.9353 | 0.9675 | 20.12 |
| 2020-Q3 | 0.5061 | 0.6721 | 0.8699 | 0.5475 | -565.15 |
| 2020-Q4 | 0.5661 | 0.7230 | 0.6572 | 0.8034 | 179.91 |
| 2021-Q1 | 0.7922 | 0.8840 | 0.8376 | 0.9360 | 72.49 |
| 2021-Q2 | 0.8128 | 0.8967 | 0.8156 | 0.9958 | 119.94 |
| 2021-Q3 | 0.4833 | 0.6517 | 0.8805 | 0.5173 | -662.31 |
| 2021-Q4 | 0.5279 | 0.6910 | 0.5910 | 0.8316 | 320.34 |
| 2022-Q1 | 0.6335 | 0.7756 | 0.6908 | 0.8842 | 142.33 |
| 2022-Q2 | 0.6190 | 0.7647 | 0.9081 | 0.6604 | -213.69 |
| 2022-Q3 | 0.5230 | 0.6868 | 0.8612 | 0.5711 | -519.60 |
| 2022-Q4 | 0.6256 | 0.7697 | 0.7454 | 0.7956 | 63.38 |
| 2023-Q1 | 0.7641 | 0.8663 | 0.8414 | 0.8926 | 38.33 |
| 2023-Q2 | 0.7147 | 0.8336 | 0.7684 | 0.9109 | 107.41 |
| 2023-Q3 | 0.6263 | 0.7702 | 0.9154 | 0.6647 | -306.30 |
| 2023-Q4 | 0.6325 | 0.7749 | 0.6465 | 0.9670 | 359.43 |
| 2024-Q1 | 0.6504 | 0.7882 | 0.6631 | 0.9713 | 215.20 |
| 2024-Q2 | 0.7352 | 0.8474 | 0.8906 | 0.8081 | -52.76 |
| 2024-Q3 | 0.5501 | 0.7097 | 0.5527 | 0.9915 | 427.37 |
| 2024-Q4 | 0.6480 | 0.7864 | 0.7055 | 0.8882 | 174.97 |
| 2025-Q1 | 0.8124 | 0.8965 | 0.8238 | 0.9833 | 118.42 |
| 2025-Q2 | 0.9298 | 0.9636 | 0.9493 | 0.9783 | 18.22 |
| 2025-Q3 | 0.5993 | 0.7495 | 0.8262 | 0.6858 | -170.04 |
| 2025-Q4 | 0.7280 | 0.8426 | 0.8655 | 0.8209 | -48.20 |
| **Mean** | **0.6956** | **0.8139** | **0.8095** | **0.8443** | **−0.47** |

*Table 7.14: Detailed Quarterly Prediction Results — Setup 2 (Attention U-Net+ConvLSTM, SeqLen=10).*

| Quarter | IoU | Dice | Precision | Recall | ∆Area (km²) |
|---------|-----|------|-----------|--------|-------------|
| 2022-Q4 | 0.6086 | 0.7567 | 0.7498 | 0.7637 | 17.46 |
| 2023-Q1 | 0.7485 | 0.8561 | 0.8325 | 0.8812 | 36.82 |
| 2023-Q2 | 0.7025 | 0.8253 | 0.7826 | 0.8729 | 66.80 |
| 2023-Q3 | 0.6164 | 0.7627 | 0.8820 | 0.6719 | -266.44 |
| 2023-Q4 | 0.6406 | 0.7810 | 0.6768 | 0.9230 | 263.79 |
| 2024-Q1 | 0.6552 | 0.7917 | 0.6769 | 0.9533 | 189.02 |
| 2024-Q2 | 0.7804 | 0.8767 | 0.9655 | 0.8028 | -96.03 |
| 2024-Q3 | 0.5669 | 0.7236 | 0.5703 | 0.9894 | 395.49 |
| 2024-Q4 | 0.6496 | 0.7876 | 0.7263 | 0.8601 | 124.49 |
| 2025-Q1 | 0.8740 | 0.9328 | 0.9362 | 0.9293 | -4.55 |
| 2025-Q2 | 0.9407 | 0.9695 | 0.9583 | 0.9809 | 14.04 |
| 2025-Q3 | 0.5745 | 0.7297 | 0.8544 | 0.6368 | -254.68 |
| 2025-Q4 | 0.7061 | 0.8278 | 0.8583 | 0.7994 | -64.14 |
| **Mean** | **0.6972** | **0.8170** | **0.8054** | **0.8511** | **32.47** |

## 7.3 Bi-Monthly Data (Higher-Frequency Temporal Analysis)

To evaluate higher-frequency satellite image prediction performance, experiments were conducted using bi-monthly composite data under two forecasting setups: (i) Setup 1: long-range bi-monthly prediction and (ii) Setup 2: medium-range bi-monthly prediction. Three deep learning architectures were evaluated: ConvLSTM, U-Net+LSTM, and Attention U-Net+ConvLSTM. Model performance was assessed using Intersection over Union (IoU), Dice coefficient, Precision, Recall, and Absolute Area Difference.

For each model, multiple sequence lengths were tested, and the best-performing configuration was selected based on mean IoU. In the current experiments, the best performance for all three architectures was obtained using a sequence length of 9. A consistent result presentation structure is maintained for all models, followed by the selection of the best-performing architecture. Detailed period-wise results for the final best model will be added in the subsequent section.

### ConvLSTM Results

Table 7.15 presents the best performance of the ConvLSTM model for both bi-monthly forecasting setups.

*Table 7.15: ConvLSTM Performance on Bi-Monthly Data.*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.6347 | 0.7732 | 0.7941 | 0.7708 | 131.44 |
| Setup 2 (medium-range) | 0.6623 | 0.7956 | 0.7979 | 0.8062 | 119.11 |

The ConvLSTM model produced the lowest performance among the three evaluated architectures for bi-monthly prediction. Setup 2 achieved slightly better IoU and Dice scores than Setup 1, indicating improved segmentation accuracy for the shorter forecasting horizon. Precision and recall remained relatively balanced across both setups, although the model showed lower overlap accuracy compared to persistence baselines. This shows that the ConvLSTM model has difficulty retaining the river boundary accurately at increased prediction frequencies.

### U-Net+LSTM Results

Table 7.16 summarizes the best results obtained using the U-Net+LSTM model.

*Table 7.16: U-Net+LSTM Performance on Bi-Monthly Data.*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.7439 | 0.8435 | 0.8658 | 0.8496 | 156.64 |
| Setup 2 (medium-range) | 0.7791 | 0.8701 | 0.8971 | 0.8691 | 160.78 |

The model that had the best performance in segmentation was U-Net+LSTM. Both models performed much better than the ConvLSTM model, with Setup 2 having the best IoU score (0.7791) and Dice coefficient score (0.8701). The good precision scores show that the river boundary was localized well, and the balanced recall shows that region detection is reliable. Compared to the persistence model, Setup 1 had a slightly better performance than the persistence model, but Setup 2 had performance similar to that of the persistence model.

### Attention U-Net+ConvLSTM Results

The best results of the Attention U-Net+ConvLSTM model are presented in Table 7.17.

*Table 7.17: Attention U-Net+ConvLSTM Performance on Bi-Monthly Data.*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.7410 | 0.8406 | 0.8806 | 0.8293 | 157.45 |
| Setup 2 (medium-range) | 0.7733 | 0.8662 | 0.9094 | 0.8459 | 147.45 |

The Attention U-Net+ConvLSTM model achieved performance comparable to U-Net+LSTM across both setups. In Setup 1, the model produced slightly lower IoU than U-Net+LSTM but higher precision, indicating stronger boundary confidence with slightly reduced spatial coverage. In Setup 2, it achieved the lowest absolute area difference among all models (147.45 km²), demonstrating improved spatial calibration and reduced prediction bias. The attention mechanism helps emphasize relevant river regions, which is particularly beneficial for high-frequency temporal prediction where subtle spatial changes become more significant.

### SwinST Results

Table 7.18 presents the best performance of the SwinST model for both bi-monthly forecasting setups.

*Table 7.18: SwinST Performance on Bi-Monthly Data.*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.6333 | 0.7718 | 0.8015 | 0.7608 | 131.99 |
| Setup 2 (medium-range) | 0.6952 | 0.8180 | 0.8404 | 0.8086 | 114.45 |

The SwinST model achieved its best results using a sequence length of 6 in both setups. Setup 2 outperformed Setup 1 across all metrics, achieving an IoU of 0.695, a Dice score of 0.818, and the lowest absolute area difference of 114.45 km², indicating stronger segmentation accuracy under the medium-range forecasting horizon. The setup one resulted in relatively lower overlap scores, which was demonstrated by the IoU of 0.633 and Dice of 0.772, as well as higher absolute area difference of 131.99 km². Thus, it was more challenging to accurately define the boundaries of the rivers within the longer prediction time frames. Moreover, precision value was higher than recall, implying the conservative behavior of the model that tends to underestimate the extent of the river, as seen from the negative values of the mean area difference parameter.

### Vision Transformer + Temporal Transformer Results

Table 7.19 presents the best performance of the Vision Transformer + Temporal Transformer model for both bi-monthly forecasting setups.

*Table 7.19: Vision Transformer + Temporal Transformer Performance on Bi-Monthly Data.*

| Setup | IoU | Dice | Precision | Recall | Abs Area Diff (km²) |
|-------|-----|------|-----------|--------|---------------------|
| Setup 1 (long-range) | 0.3805 | 0.5487 | 0.5461 | 0.5771 | 203.91 |
| Setup 2 (medium-range) | 0.3421 | 0.5079 | 0.4723 | 0.5711 | 243.07 |

As far as the Vision Transformer + Temporal Transformer is concerned, this model demonstrated the worst performance among all other tested architectures for bi-monthly prediction, underperforming the persistence baseline and all other deep learning models according to all metrics. Setup 1 yielded the highest score at sequence length of 9, with IoU being 0.381 and Dice 0.549, whereas Setup 2 achieved the best performance with sequence length 6, providing IoU 0.342 and Dice 0.508. In contrast with the previous models, in this particular case, Setup 2 did not outperform Setup 1. Precision and recall values proved to be low and imbalanced for both setups, showing that the model failed to distinguish river extents. Furthermore, the large positive mean area difference values suggest a tendency to systematically overestimate river extent, in contrast to the conservative underestimation observed in the other architectures. The high absolute area differences further confirm that the Vision Transformer + Temporal Transformer architecture, in its current form, is not well suited for river extent prediction at bi-monthly temporal resolution.

### Best Model Selection

Among all five evaluated architectures, the U-Net+LSTM model is selected as the best-performing model for bi-monthly river prediction. It achieved the highest overall segmentation accuracy, with the best Setup 2 performance of IoU = 0.7791 and Dice = 0.8701, outperforming ConvLSTM, Attention U-Net+ConvLSTM, SwinST, and Vision Transformer + Temporal Transformer.

Although the Attention U-Net+ConvLSTM model produced competitive results and showed slightly better precision and strong spatial calibration through lower absolute area difference, its IoU and Dice scores remained slightly below those of U-Net+LSTM. Similarly, the SwinST model demonstrated moderate improvement over ConvLSTM, particularly in Setup 2, but its overall overlap accuracy was still substantially lower than U-Net+LSTM. The Vision Transformer + Temporal Transformer model showed the weakest performance among all architectures and was unable to produce reliable river boundary prediction at bi-monthly temporal resolution.

These results indicate that as temporal resolution increases from yearly to quarterly and bi-monthly prediction, accurate spatial feature extraction becomes increasingly important. The encoder–decoder structure of U-Net+LSTM enables better preservation of fine river boundaries while maintaining temporal consistency, making it the most reliable architecture for higher-frequency river forecasting.

## 7.4 Long-Term Prediction

*Figure 7.1: Long-term combined morphological risk for 2026–2040.*

**Live Demo Link:** https://kaoserahamed.github.io/Thesis/

The long-term prediction experiment was performed using the final Attention U-Net + ConvLSTM model with a sequence length of five years, trained on the full 1987–2025 dataset. Model optimization was carried out for a maximum of 200 epochs with early stopping (patience = 20), and training converged at epoch 107. The best model checkpoint achieved a validation loss of 0.2298, a validation Dice coefficient of 0.8057, and a validation IoU of 0.7026. These results indicate a good level of segmentation agreement on the validation set and provide a reasonable basis for multi-step forecasting. In addition, a reconstruction-based sanity check on known years (2016–2025) yielded a mean IoU of 0.6543 and a mean Dice score of 0.7907, confirming that the model was able to reproduce historical river extent with acceptable consistency, although some years showed noticeable area overestimation. The forecast of 2026–2040 exhibits a steadily increasing trend of the river extent, which implies the ongoing morphology process throughout the whole forecast horizon. In contrast to the baseline area of 517.6173 km², the river area grew considerably to 671.3261 km² in 2026 and eventually reached 807.6276 km² in 2040.

As a result, an increment of 290.0102 km², or 56.03%, was recorded against the baseline situation. An evident spike in the area was observed at the beginning forecast phase (2025–2026), whereas in the further year increments were lower but still positive, which might imply the ongoing channel readjustment process.

Based on temporal statistics, one can observe that the expansion process did not have a steady pace throughout the forecast horizon. While the early forecast years exhibited rather quick growth, the annual increment became noticeably lower after about 2031. In particular, a decrease from 19.3778 km² (between 2028 and 2029) to 4.5981 km² (between 2039 and 2040) was noted.

The analysis of change detection indicates that the development scenario is dominated by transitions resembling accretion processes. The total amount of predicted changes till 2040 in comparison to 2025 were equal to 35.4712 km² for erosion, 325.4815 km² for accretion, and 482.1461 km² for a stable river corridor. The significant predominance of the accretion process in comparison to erosion process shows that the future developments are characterized by occupation of areas located outside of the 2025 river class while there is no abandonment of the current channel footprint. Erosion was insignificant and nearly constant during the period considered, whereas accretion steadily increased from 186.5524 km² in 2026 to 325.4815 km² in 2040.

From the geomorphological point of view, such behavior suggests the lateral river channel expansion and migration to neighboring floodplain zones. Nevertheless, such terms should be used carefully, since in the case of mask-based analysis "accretion" and "erosion" denote transition of the river class in space in comparison to the baseline map and not the actual field measurements of sedimentation/deposition/erosion process.

The long-term risk map gives the most convincing geographical explanation of the above changes. High probability zones of change are mostly located at the active channel margins and adjacent floodplain zones and not at any place of the study area. It means that future instability will continue to stay within a geomorphologically active corridor. High and very high-risk zones represent areas of change that are consistently predicted over the forecast horizon, implying future certainty about channel occupation or abandonment. These zones are especially significant for infrastructure safety, agriculture, and settlement management.

The geographical distribution of risk zones proves convincingly the temporal statistics. The moderate and high-risk zones near the channel margins match almost perfectly the increasing river area over time. The very high risk zones denote persistent instability and recurrent adjustment, and low-risk zones should mark the boundaries of the active corridor. The cumulative risk assessment is much more valuable than one-year prediction because it shows the tendency of the river to rearrange geographically.

Nevertheless, one should take into account the fact that the reconstruction performed by the model overestimated the area of the river, showing a mean value of 650.3061 km² for the predicted river area and 558.4750 km² for the actual river area, which means that there was an average positive bias of 91.8311 km². In other words, the absolute extent of the river expansion might be slightly overstated by the model. Thus, the model can better predict the relative areas of future instability rather than the future river footprint.

In conclusion, it is possible to state that the study river will continue to adjust morphologically between 2026–2040 years; however, this process will be associated with the current river course only. Taking into consideration the expected increase in the size of the river and the dominance of the processes of accretion, as well as high risk zone localization, it becomes clear that there is a great probability of further channel widening.

## 7.5 Short-Term Bi-Monthly Prediction

**Live Demo Link:** https://kaoserahamed.github.io/Thesis/

The short-term bi-monthly prediction experiment was conducted using a U-Net combined with a ConvLSTM2D temporal module, trained on the full bi-monthly dataset with a fixed input sequence length of nine consecutive bi-monthly periods. Each input frame was extended with two additional channels encoding sinusoidal and cosinusoidal seasonal position, allowing the model to distinguish between the six intra-annual periods without relying solely on the spatial content of the masks. Model optimisation was carried out for a maximum of 200 epochs with early stopping (patience = 25) and a minimum improvement threshold of 10⁻⁴. Training converged at epoch 39, with the best checkpoint achieving a validation loss of 0.2219, a validation Dice coefficient of 0.8381, and a validation IoU of 0.7463. These values indicate satisfactory segmentation agreement on the held-out temporal validation split and provide an acceptable basis for multi-step bi-monthly forecasting.

A reconstruction-based sanity check was performed on the 18 most recent known bi-monthly periods spanning 2023 P1 through 2025 P6. The model reproduced historical river extents with a mean IoU of 0.6839 ± 0.0717 and a mean Dice score of 0.8102 ± 0.0525, alongside a mean precision of 0.8008 ± 0.0809 and a mean recall of 0.8376 ± 0.1062. The mean predicted area across the sanity check periods was 766.65 km² against a mean observed area of 741.34 km², yielding an average positive bias of 25.30 km². The slight overestimation is significantly lower compared to the one in the annual cycle and implies that two monthly time resolution gives more information within one year cycle, minimizing systematic bias. The worst reconstructed sequence was the one for 2023 P4 (July–August), which had the IoU score of just 0.5178 and a considerable area underestimation value of −551.10 km². This corresponds to the period of peak monsoonal flooding, when usually the largest river extents are formed.

### Bi-Monthly Forecast: 2026 P1–2028 P6

18 sequential bi-monthly periods were used for autoregressive forecasting, amounting to three full years of predictions, starting from the baseline period of 2025 P6 (Nov-Dec) with an observed area of 934.07 km². For this purpose, a rolling input window was started using the latest nine periods (2024 P4-2025 P6), and raw sigmoid probability maps were supplied to the next iteration instead of binarized binary masks.

As can be seen from the forecast results, there is a clear three-phase structure in time. At the first forecasting year (2026), the mean predicted extent of the river has decreased to 902.69 km², which means that the mean deviation from the baseline is −31.38 km². During the year, the minimal extent was observed at 2026 P3 (May–Jun) with 828.18 km² extent, which is −105.89 km² (−11.34%) compared to the baseline level, while the maximal extent has reached 981.13 km² at 2026 P5 (Sep–Oct), that is 5.04% above baseline. The first stage of contraction and then extension is related to the carrying out of the dry-to-wet seasonal transition pattern in the seed sequence by the model. At the second forecasting year (2027), the mean extent increased to 983.98 km² with a mean deviation of +49.91 km² above baseline, with the peak reaching 1050.59 km² at 2027 P5 (Sep–Oct), equivalent to a +12.47% expansion. By the third forecast year (2028) the mean area rose further to 1020.10 km², and the final predicted period 2028 P6 (Nov–Dec) recorded an extent of 1054.39 km², a net increase of 120.32 km² (+12.88%) above the 2025 P6 baseline.

The slow increase through the years of forecasting as opposed to an immediate jump shows that the autoregressive model has accounted for the evolutional morphological adaptation rather than the statistical attraction into a fixed state. The stabilising behaviour observed in the earlier identical-output run is absent here, confirming that the seasonal encoding and probability-feedback mechanism were effective in maintaining temporal diversity across all 18 prediction steps.

### Seasonal Patterns

Monthly aggregation over all three forecasting years exhibits a clear and geophysically sensible seasonality trend. The minimum area mean occurs at P3 (May–Jun) at 918.25 km², representing the late pre-monsoon period during which the river reaches its narrowest form. Maximum mean area is predicted at P5 (Sep–Oct) at 1034.27 km², occurring during the peak post-monsoon period after the flow has attained its widest configuration. The associated seasonality amplitude is 116.02 km², which is consistent geographically with the expected value for a major braided or meandering system subjected to heavy monsoonal forcing.

Monthly aggregation over monsoon-affected periods (P4 Jul–Aug and P5 Sep–Oct) against non-monsoon periods results in a monsoon mean of 1004.54 km² versus non-monsoon mean of 951.12 km². Consequently, monsoon expansion effect is +53.42 km², or +5.62% expansion over the non-monsoon mean area. This level of seasonality contrast is realistic for South Asian river systems and provides empirical validation that the seasonal encoding has effectively conveyed intra-annual periodicity from training to forecasting phase without additional hydrological inputs.

### Change Detection Analysis

Change detection was calculated for each of the 18 forecast periods in relation to the 2025 P6 baseline. On average, the model detected 188.97 km² of erosion type changes (river boundary retreat), 223.83 km² of accretion type changes (new areas occupied by river), and 34.86 km² of net positive change per period. The largest erosion was detected for the 2028 P3 period (209.51 km²) - the dry season when the channel boundary loses its extent in the baseline period, while the largest accretion was detected for the 2028 P5 period (333.62 km²) – the late monsoon period characterized by maximum lateral inundation. Such a systematic difference, when erosion reaches its maximum value during transition periods and accretion is the dominant process during and after the peak discharge periods is consistent with the flood hysteresis.

The last forecast period 2028 P6 is characterized by 201.54 km² of erosion, 321.86 km² of accretion and 732.53 km² of stable river area left from the 2025 P6 baseline. Since the ratio between the accretion and erosion in the last period is approximately 1.60:1, one can conclude that over the three-year horizon the river is going to expand more than it contracts relative to the baseline footprint. Stable area constitutes 78.4% of the baseline extent by the final period. This suggests that the core channel geometry remains broadly consistent while the margins adjust.

### Geomorphological Interpretation

The bi-monthly prediction approach affords greater temporal resolution compared to annual forecasts and is able to resolve dynamics not seen in annual-scale models. The bi-monthly prediction model accurately reflects the expansion/contraction pattern caused by the monsoons, along with the relative asymmetry in wet- and dry-season channel sizes. The increasing trend of mean annual area for 2026, 2027, and 2028 implies a superposition of long-term trends of gradual expansion of the channel area on top of the seasonal cycle, resulting from continuous channel migration and floodplain occupation which cannot be reversed during the entire dry season.

The high-risk transition zones occur near the channel margins exposed to monsoon flooding in periods P4 and P5, where the likelihood of accretion is highest. The low-risk zones are represented by the inner stable part of the river channel, which remains flooded for all six bi-monthly periods. The moderate-risk zones in the outer edges of post-monsoon channel extent include areas that get temporarily flooded and dry out; these are the regions with the greatest risk of being reclassified as land on an incremental basis.

---

# 8. Discussion

The river extent prediction results are really interesting. They show that hybrid CNN-LSTM architectures work better than transformer-based models for predicting river extent over a year, every quarter, and every two months. This tells us that to predict river extent accurately we need to preserve the details of the river boundaries and also learn how the river changes over time. Some models like ConvLSTM, U-Net+LSTM and Attention U-Net+ConvLSTM worked well because they used layers to capture the local details of the river and recurrent layers to model how the river changes over time.

When we looked at predicting river extent over a year, the Attention U-Net+ConvLSTM model did the job. It was especially good at predicting river extent over a 10-year period with the IoU and Dice score. This shows that using attention mechanisms helps with long-term predictions by focusing on the parts of the river and reducing noise. Although U-Net+LSTM also did a job, the Attention U-Net+ConvLSTM model handled the uncertainty of long-term predictions better. We also found that predicting river extent over a period of 5 years generally gave us more accurate results, which means that shorter prediction periods give us more precise results and less uncertainty. Predicting river extent every quarter was harder because of the changes that happen during seasons. Cloud cover and flooding made it harder to see the river boundaries. Made predictions more complex. Again the Attention U-Net+ConvLSTM model did the best job of balancing accuracy and precision. We noticed that the model did better in some quarters than others, which shows that it is sensitive to changes. Even with these fluctuations the model still did a good job overall and did not have a lot of bias in its predictions.

For bi-monthly predictions the U-Net+LSTM model did the best job, with the highest IoU and Dice score. When we are predicting river extent at a temporal resolution, the small details of the river boundaries become more important than looking at long-term changes. The U-Net model preserved these details better and the LSTM model helped with short-term consistency.

The transformer-based models like ViT, ViViT and Vision Transformer + Temporal Transformer did not do well in our experiments. They had low IoU and Dice scores and often overestimated the river extent. This suggests that pure transformer models struggle to capture the details of the river when we do not have a lot of training data. We also found that using IoU and Dice scores is not enough to evaluate the models. Some models had overlap scores but still had large errors in their predictions. Since river monitoring depends on measurements we need to consider both overlap and error when evaluating the models. Overall our results show that the performance of the models depends on the time period we are predicting and the level of detail we need. The Attention U-Net+ConvLSTM model is best for quarterly predictions while the U-Net+LSTM model is best for bi-monthly predictions. This tells us that there is no one model that works best for all cases and we need to choose the model based on the balance between long-term changes and precise boundary preservation. River extent prediction results show that hybrid CNN-LSTM architectures like Attention U-Net+ConvLSTM and U-Net+LSTM are better than transformer-based models for river extent prediction.

---

# 9. Future Works and Research Directions

This study opens up directions worth exploring in follow-up work both in terms of improving the models and making them more useful in real-life situations.

One immediate step is to focus on the Padma channel instead of looking at the entire scene. This would help the models ignore noise from ponds and minor rivers and would likely make both training and predictions better.

The transformer-based architectures, like Swin Transformer and Vision Transformer did not perform well in this study. This is probably because we did not have data, rather than a problem with the architecture itself. If we had training samples or if we pre-trained these models on larger datasets before fine-tuning them on the Padma time series they could become more competitive. It would be worth trying them as we collect more data.

On the data side if we could get Sentinel-2 imagery for the whole study period not just after 2014 we could increase the spatial resolution from 60m to 10m and see more details. Sentinel-1 SAR data can help during monsoon months when optical sensors are often blocked by clouds. This would improve data quality.

The current framework only looks at the Padma River. If we extend it to rivers in Bangladesh like the Jamuna or Meghna we can see how well the models work in different situations and make the system more useful for the whole country.

Finally the long-term predictions we made (2026-2040) show where the risk zones are. They do not take into account the social and economic context. If we overlay the erosion probability maps, with population density, agricultural land-use data and critical infrastructure planners can estimate the risk and prioritize areas that need help. This is a next step that does not require changing the core model.

---

# 10. Conclusion

This study looked at whether it's possible to use deep spatiotemporal learning to predict the changes in the shape of the Padma River using only data from satellites. The results show that this method can work. There are some important things to consider when choosing a model and predicting what will happen in the future.

We created a 38-year record of where the water's in the Padma River from 1987 to 2025 using images from Landsat and Sentinel satellites that were processed using Google Earth Engine. We used a method called MNDWI to find the water and filled in any missing information using BiConvLSTM, which gave us a consistent set of data that we could use to look at changes over time.

We tried out five models to see which one worked best: ConvLSTM, U-Net with LSTM, Attention U-Net with ConvLSTM, Swin Transformer and Vision Transformer. The models that combined CNN and LSTM worked better than the ones that used transformers, which had trouble learning the shape of the river without enough training data. When we looked at predicting what would happen every year and every quarter the Attention U-Net with ConvLSTM model worked the best with an accuracy of 0.7005 and 0.8236. When we looked at predicting what would happen every two months the U-Net with LSTM model worked the best with an accuracy of 0.779 and 0.870.

We used this model to predict what the Padma River will look like from 2026 to 2040. Found that the river will get bigger by about 290 square kilometers compared to 2025. The river will keep getting larger. It would slow down after 2031. We also made maps that show which parts of the river are at risk of erosion, which can help people plan for the future and get ready for disasters.

Overall these results show that using learning to predict changes in the shape of the Padma River is a good idea and can be done using only free satellite data and open platforms. For people living near the Padma River in Bangladesh being able to predict what will happen a years in advance can make a big difference, in planning and preparing for the future. The Padma River is a part of the study and the results of the study can help people understand the Padma River better.

---

# References

[1] P. Lemenkova, "Deep learning methods of satellite image processing for monitoring of flood dynamics in the ganges delta, bangladesh," *Water*, vol. 16, no. 8, p. 1141, 2024.

[2] J. Freihardt et al., "Assessing riverbank erosion in bangladesh using time series of sentinel-1 radar imagery in the google earth engine," *Natural Hazards and Earth System Sciences*, vol. 23, pp. 751–768, 2023.

[3] G. M. M. Alam et al., "Exploring impacts and livelihood vulnerability of riverbank erosion hazard among rural household along the river padma of bangladesh," *Environmental Systems Research*, vol. 6, p. 25, 2017.

[4] M. N. Islam et al., "Socioeconomic impacts and migration dynamics of riverbank erosion," *npj Climate Action*, vol. 4, p. 6, 2025.

[5] T. K. Das et al., "Impact of riverbank erosion: A case study," *Australasian Journal of Disaster and Trauma Studies*, vol. 21, no. 2, pp. 73–81, 2017.

[6] D. Sarkar et al., "Assessing riverbank erosion and livelihood resilience using traditional approaches in northern bangladesh," *Sustainability*, vol. 14, no. 4, p. 2348, 2022.

[7] D. V. Binh et al., "A novel method for river bank detection from landsat time series (srbed)," *Remote Sensing*, vol. 12, no. 20, p. 3298, 2020.

[8] H. Fan et al., "Using long short-term memory networks for river flow prediction," *Hydrology Research*, vol. 51, no. 6, pp. 1358–1376, 2020.

[9] A. Dehghani et al., "Comparative evaluation of lstm, cnn, and convlstm for spatiotemporal forecasting tasks," *Ecological Informatics*, vol. 75, p. 101984, 2023.

[10] M. S. Rafat et al., "From pixels to people: Satellite-based mapping and quantification of riverbank erosion and lost villages in bangladesh," arXiv preprint arXiv:2510.17198, 2025.

[11] T. Langhorst et al., "Global observations of riverbank erosion and accretion from landsat imagery," *Journal of Geophysical Research: Earth Surface*, vol. 128, no. 2, p. e2022JF006774, 2023.

[12] P. K. Langat et al., "Monitoring river channel dynamics using remote sensing and gis techniques," *Geomorphology*, vol. 325, pp. 92–106, 2019.

[13] M. N. Islam and M. A. Haque, "River channel migration: A remote sensing and gis analysis," in *Proceedings of ESA Living Planet Symposium*, ser. ESA SP-686, Bergen, Norway, June–July 2010.

[14] P. V. Raju et al., "Spatio-temporal analysis of riverbank changes using remote sensing and geographic information system," *Ecological Informatics*, vol. 81, p. 102628, 2024.

[15] J. Freihardt, "Sentinel-1 time series for jamuna riverbank erosion (methods, gee scripts)," ETH Zurich, Technical report, 2023.

[16] T. Tha, T. Piman, D. Bhatpuria, and P. Ruangrassamee, "Assessment of riverbank erosion hotspots along the mekong river in cambodia using remote sensing and hazard exposure mapping," *Water*, vol. 14, no. 19, p. 2995, 2022.

[17] O. R. Saha, "Riverbank migration and island dynamics at the padma confluence: multi-sensor analysis," *Geomorphology*, vol. 448, p. 109043, 2025.

[18] M. A. Haque, "Change detection of jamuna river and its impact on local land use and settlement," *Journal of Urban and Regional Analysis*, vol. 15, no. 2, pp. 145–162, 2023.

[19] S. M. Ritu, S. K. Sarkar, and H. Zonaed, "Prediction of padma river bank shifting and its consequences on lulc changes," *Ecological Indicators*, vol. 156, p. 111104, 2023.

[20] C. Muzahid, S. Popy, R. Islam, M. S. A. Emon, M. S. Reja, M. M. Rahman, J. Hoque, M. G. Rabbani, and S. Raiyan, "Quantifying river bank erosion and accretion patterns along the gorai river in kushtia, bangladesh: A geospatial analysis utilizing gis and remote sensing techniques," *Journal of Geographic Information System*, vol. 16, pp. 70–88, 2024.

[21] Z. Ren, "A large-scale riverbank erosion risk assessment model with remote sensing inputs," *Applied Geography*, vol. 162, p. 103156, 2024.

[22] K. Yuan et al., "Deep-learning-based multispectral satellite image segmentation for water body detection," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 14, pp. 7422–7434, 2021.

[23] Y. Li et al., "Assessing the performance and interpretability of the cnn-lstm-attention model for daily streamflow forecasting in typical basins of the eastern qinghai-tibet plateau," *Scientific Reports*, vol. 15, p. 260, 2025.

[24] S. Ghosh et al., "Water body extraction from high spatial resolution remote sensing images based on enhanced u-net and multi-scale information fusion," *Scientific Reports*, vol. 14, p. 16192, 2024.

[25] B. DeVries et al., "Rapid and robust monitoring of flood events using sentinel-1 and landsat data on the google earth engine," *Remote Sensing of Environment*, vol. 240, p. 111664, 2020.

[26] M. A. Clement et al., "Visualisation of flooding along an unvegetated, ephemeral river using google earth engine," *Journal of Environmental Management*, vol. 277, p. 111338, 2020.

[27] P. Tripathy and A. Mallick, "Applying google earth engine for flood mapping and monitoring in the downstream provinces of mekong river," *Results in Engineering*, vol. 14, p. 100432, 2022.

[28] N. Sazib et al., "Google earth engine for large-scale flood mapping using sar data and impact assessment on agriculture and population of ganga-brahmaputra basin," *Sustainability*, vol. 14, no. 7, p. 4210, 2022.

[29] X.-H. Le et al., "Streamflow prediction using an integrated methodology based on convolutional neural network and long short-term memory networks," *Scientific Reports*, vol. 11, p. 17429, 2021.

[30] Z. Xiang, J. Yan, and I. Demir, "Hybrid cnn-lstm models for river flow prediction," *Water Supply*, vol. 22, no. 5, pp. 4902–4919, 2022.

[31] A. Horvath et al., "Water level prediction using long short-term memory neural network model for a lowland river: a case study on the tisza river, central europe," *Environmental Sciences Europe*, vol. 35, p. 92, 2023.

[32] N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore, "Google earth engine: Planetary-scale geospatial analysis for everyone," *Remote Sensing of Environment*, vol. 202, pp. 18–27, 2017.

[33] M. Sarafanov, E. Kazakov, N. O. Nikitin, and A. V. Kalyuzhnaya, "A machine learning approach for remote sensing data gap-filling with open-source implementation: An example regarding land surface temperature, surface albedo and ndvi," *Remote Sensing*, vol. 12, no. 23, p. 3865, 2020.

[34] Z. Tang, H. Adhikari, P. K. E. Pellikka, and J. Heiskanen, "Impact of preprocessing on tree canopy cover modelling: Does gap-filling of landsat time series improve modelling accuracy?" *Frontiers in Remote Sensing*, vol. 3, p. 936194, 2022.

[35] S. P. Raya and J. K. Udupa, "Shape-based interpolation of multidimensional objects," *IEEE Transactions on Medical Imaging*, vol. 9, no. 1, pp. 32–42, 1990.

[36] F. Susanto, P. De Souza, and J. He, "Spatiotemporal interpolation for environmental modelling," *Sensors*, vol. 16, no. 8, p. 1245, 2016.

[37] Q. Liu, F. Zhou, R. Hang, and X. Yuan, "Bidirectional-convolutional lstm based spectral-spatial feature learning for hyperspectral image classification," *Remote Sensing*, vol. 9, no. 12, p. 1330, 2017.

[38] M. I. J. Putra, E. Irwansyah, A. Gustini, M. C. Harist, R. A. Abdurrahman, Supriatna, and V. Alexander, "ConvLSTM for pixel-level shoreline retreat spatiotemporal forecasting," in *The 1st Workshop on Monitoring the World through an Imperfect Lens*, 2026. [Online]. Available: https://openreview.net/forum?id=JGlX6wwgIs

[39] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, "Video swin transformer," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 3202–3211.

[40] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," arXiv preprint arXiv:2010.11929, 2021.

[41] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Lučić, and C. Schmid, "ViViT: A video vision transformer," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 6836–6846.

---

*Document converted from `archive/reports_root/Predefence_report (1).pdf`. Last Updated: September 22, 2026.*
