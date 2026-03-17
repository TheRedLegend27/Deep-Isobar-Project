# Module: City Universe

Purpose

Define cities traded by Deep Isobar and configure their profiles.

City Profiles Include

station id
forecast bias correction
variance multiplier
KDE bandwidth

Example

Chicago

station: KORD
variance_multiplier: 1.2

Example City Universe

New York
Chicago
Dallas
Houston
Phoenix
Miami
Atlanta
Denver
Las Vegas
Seattle

Outputs

city profile configuration

Tasks

build city profile loader
store profiles in YAML
