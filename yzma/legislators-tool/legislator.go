package main

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

type Legislator struct {
	BioguideID         string      `json:"bioguide_id"`
	Fullname           string      `json:"full_name"`
	Address            string      `json:"address"`
	State              string      `json:"state"`
	District           string      `json:"district"`
	Party              string      `json:"party"`
	Type               string      `json:"legtype"`
	Phone              string      `json:"phone"`
	ContactForm        string      `json:"contact_form"`
	GovTrackProfile    string      `json:"govtrack_profile"`
	OpensecretsSummary string      `json:"opensecrets"`
	URL                string      `json:"uel"`
	Offices            []Office    `json:"offices"`
	Committees         []Committee `json:"committees"`
}

type Office struct {
	Address  string `json:"address"`
	Building string `json:"building"`
	City     string `json:"city"`
	State    string `json:"state"`
	Zip      string `json:"zip"`
	Fax      string `json:"fax"`
	Hours    string `json:"hours"`
	Phone    string `json:"phone"`
	Suite    string `json:"suite"`
}

type Committee struct {
	Name                      string `json:"name"`
	CommitteeType             string `json:"committee_type"`
	CommitteeName             string `json:"committee_name"`
	CommitteeSubCommitteeName string `json:"committee_subcommittee_name"`
	Party                     string `json:"party"`
	Title                     string `json:"title"`
	Rank                      string `json:"rank"`
	Chamber                   string `json:"chamber"`
}

func GetLegislator(fullname string) (*Legislator, error) {
	db, err := sql.Open("sqlite3", "static/legislators.db")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	defer db.Close()

	var first_name string
	var last_name string
	var full_name string
	var address string
	var state string
	var district string
	var legtype string
	var party string
	var url string
	var contact_form string
	var phone string
	var bioguide_id string
	var govtrack_profile string
	var opensecrets string

	full_name = fullname

	// handle middle initial with period. split so we can query on
	// first_name and last_name.
	fullname_string := strings.Split(full_name, " ")
	if len(fullname_string) == 3 {
		first_name = fullname_string[0]
		last_name = fullname_string[2]
	} else if len(fullname_string) >= 2 {
		first_name = fullname_string[0]
		last_name = fullname_string[1]
	} else {
		return nil, fmt.Errorf("invalid name format: %s", fullname)
	}

	sql_query := `
	SELECT first_name, 
		last_name, 
		full_name, 
		address,  
		state, 
		district, 
		party, 
		type as legtype, 
		phone,  
		contact_form, 
		url, 
		bioguide_id,
		govtrack_id as govtrack_profile,
		opensecrets_id as opensecrets
	FROM legislators 
	WHERE first_name = ? 
	AND last_name = ?`

	err = db.QueryRow(sql_query, first_name, last_name).Scan(&first_name, &last_name, &full_name, &address, &state, &district, &party, &legtype, &phone, &contact_form, &url, &bioguide_id, &govtrack_profile, &opensecrets)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("no legislator found for %s", fullname)
	} else if err != nil {
		return nil, fmt.Errorf("query row scan: %w", err)
	}

	offices, err := GetLegislatorOffices(bioguide_id)
	if err != nil {
		return nil, err
	}

	committees, err := GetLegislatorCommittees(bioguide_id)
	if err != nil {
		return nil, err
	}

	legistator := fmt.Sprintf("%s %s %s (%s)", party, legtype, full_name, state)
	govtrack_url := fmt.Sprintf("https://www.govtrack.us/congress/members/%s_%s/%s", strings.ToLower(first_name), strings.ToLower(last_name), govtrack_profile)
	opensecrets_url := fmt.Sprintf("https://www.opensecrets.org/members-of-congress/%s-%s/summary?cid=%s", strings.ToLower(first_name), strings.ToLower(last_name), opensecrets)

	return &Legislator{
		BioguideID:         bioguide_id,
		Fullname:           legistator,
		Type:               legtype,
		Address:            address,
		District:           district,
		GovTrackProfile:    govtrack_url,
		OpensecretsSummary: opensecrets_url,
		Phone:              phone,
		URL:                url,
		ContactForm:        contact_form,
		Offices:            offices,
		Committees:         committees,
	}, nil
}

func GetLegislatorOffices(bioguide_id string) ([]Office, error) {
	db, err := sql.Open("sqlite3", "static/legislators.db")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	defer db.Close()

	sql_query_offices := `
	SELECT address, 
		building, 
		city, 
		fax, 
		hours, 
		phone, 
		state, 
		suite, 
		zip
	FROM legislators_district_offices 
	WHERE bioguide = ? 
	`

	var offices []Office
	rows, err := db.QueryContext(
		context.Background(), sql_query_offices, bioguide_id,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {

		var office Office

		if err := rows.Scan(
			&office.Address, &office.Building, &office.City, &office.Fax, &office.Hours, &office.Phone, &office.State, &office.Suite, &office.Zip,
		); err != nil {
			return nil, err
		}
		offices = append(offices, office)
	}

	return offices, nil
}

func GetLegislatorCommittees(bioguide_id string) ([]Committee, error) {
	db, err := sql.Open("sqlite3", "static/legislators.db")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	defer db.Close()

	sql_query_committees := `
	SELECT name, 
		committee_type,
		committee_name, 
		committee_subcommittee_name, 
		party, 
		title, 
		rank, 
		chamber
	FROM legislators_committee_membership 
	WHERE bioguide = ? 
	`

	var committees []Committee
	rows, err := db.QueryContext(
		context.Background(), sql_query_committees, bioguide_id,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {

		var committee Committee

		if err := rows.Scan(
			&committee.Name, &committee.CommitteeType, &committee.CommitteeName, &committee.CommitteeSubCommitteeName, &committee.Party, &committee.Title, &committee.Rank, &committee.Chamber,
		); err != nil {
			return nil, err
		}
		committees = append(committees, committee)
	}

	return committees, nil
}
