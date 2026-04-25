package main

import (
	"database/sql"
	"fmt"
	"log"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

type Legislator struct {
	Fullname           string `json:"full_name"`
	Address            string `json:"address"`
	State              string `json:"state"`
	District           string `json:"district"`
	Party              string `json:"party"`
	Type               string `json:"legtype"`
	Phone              string `json:"phone"`
	ContactForm        string `json:"contact_form"`
	GovTrackProfile    string `json:"govtrack_profile"`
	OpensecretsSummary string `json:"opensecrets"`
	URL                string `json:"uel"`
}

func GetLegislator(fullname string) (*Legislator, error) {
	db, err := sql.Open("sqlite3", "static/legislators.db")
	if err != nil {
		log.Fatal(err)
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
	var govtrack_profile string
	var opensecrets string

	full_name = fullname

	// handle middle initial with period. split so we can query on
	// first_name and last_name.
	fullname_string := strings.Split(full_name, " ")
	if len(fullname_string) == 3 {
		first_name = fullname_string[0]
		last_name = fullname_string[2]
	} else {
		first_name = fullname_string[0]
		last_name = fullname_string[1]
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
		govtrack_id as govtrack_profile,
		opensecrets_id as opensecrets
	FROM legislators 
	WHERE first_name = ? 
	AND last_name = ?`

	err = db.QueryRow(sql_query, first_name, last_name).Scan(&first_name, &last_name, &full_name, &address, &state, &district, &party, &legtype, &phone, &contact_form, &url, &govtrack_profile, &opensecrets)
	if err == sql.ErrNoRows {
		log.Println("No legislator found")
	} else if err != nil {
		log.Fatal(err)
	}

	legistator := fmt.Sprintf("%s %s %s (%s)", party, legtype, full_name, state)
	govtrack_url := fmt.Sprintf("https://www.govtrack.us/congress/members/%s_%s/%s", strings.ToLower(first_name), strings.ToLower(last_name), govtrack_profile)
	opensecrets_url := fmt.Sprintf("https://www.opensecrets.org/members-of-congress/%s-%s/summary?cid=%s", strings.ToLower(first_name), strings.ToLower(last_name), opensecrets)

	return &Legislator{
		Fullname:           legistator,
		Type:               legtype,
		Address:            address,
		District:           district,
		GovTrackProfile:    govtrack_url,
		OpensecretsSummary: opensecrets_url,
		Phone:              phone,
		URL:                url,
		ContactForm:        contact_form,
	}, nil
}
