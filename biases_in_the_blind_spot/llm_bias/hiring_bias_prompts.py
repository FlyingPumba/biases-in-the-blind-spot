import re


def remove_resume_title(resume):
    title = re.findall(r"\b[A-Z]+\b", resume[:100])
    for t in title:
        resume = resume.replace(t, "")
    return resume


def fix_inconsistencies_in_resume(
    base_resume: str,
    name: str,
    gender: str,
    race: str,
    pronouns: str,
    email: str,
    job_category: str,
) -> str:
    """
    Fixes inconsistencies in the resume string based on the provided applicant details.
    This function addresses common errors found in generated resumes, such as conflicting
    names, pronouns, and gendered terms.

    ASSUMPTIONS:
    - The 'name' parameter is in "FirstName LastName" format.
    - The 'gender' parameter is either "Male" or "Female".
    """
    first_name = name.split()[0]

    # 1. Fix conflicting names
    base_resume = re.sub(
        r"Christopher\s+Townes\s+Summary", f"{name} Summary", base_resume
    )
    base_resume = re.sub(r"William\s+Hale", name, base_resume)
    base_resume = re.sub(r"Bill\s+Hale", name, base_resume)
    base_resume = re.sub(r"\bBill\b", first_name, base_resume)
    base_resume = re.sub(r"Christopher\s+Townes", name, base_resume)
    base_resume = re.sub(r"\bWilliam\b", first_name, base_resume)

    # 2. Fix inconsistent LinkedIn URL
    name_parts = name.split()
    linkedin_id = f"{name_parts[0].lower()}{name_parts[-1].lower()}"
    base_resume = re.sub(
        r"(linkedin.com/in/)[a-zA-Z0-9]+", rf"\\1{linkedin_id}", base_resume
    )

    if gender == "Female":
        # 3. Fix gendered terms for female applicants
        base_resume = re.sub(
            r"Delta\s+Sigma\s+Phi\s+Fraternity\s+Brother",
            "Delta Sigma Theta Sorority Sister",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Fraternity\s+Brother", "Sorority Sister", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(r"\bhis\b", "her", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bhe\b", "she", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bhim\b", "her", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Male\s+Athlete", "Female Athlete", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(r"\bactor\b", "actress", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"master\s+of\s+ceremonies",
            "mistress of ceremonies",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"Mr\.", "Ms.", base_resume)
        base_resume = re.sub(
            r"Outstanding\s+Young\s+Men\s+of\s+America",
            "Outstanding Young Women of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bguy\b", "gal", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bman\b", "woman", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"\bFootball\b", "Soccer", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(
            r"Boy\s+Scouts\s+of\s+America",
            "Girl Scouts of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Technology\s+Professionals\s+of\s+Wisconsin,\s+Inc\.",
            "Women in Technology Wisconsin, Inc.",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Arizona\s+Business\s+and\s+Professional\s+Association",
            "Arizona Business and Professional Women",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bTOMER\b", first_name, base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Jusitn:", f"{first_name}:", base_resume, flags=re.IGNORECASE
        )

    elif gender == "Male":
        # 4. Fix gendered terms for male applicants
        base_resume = re.sub(
            r"\bher\b(?!\s[a-zA-Z])", "him", base_resume, flags=re.IGNORECASE
        )  # Objective pronoun
        base_resume = re.sub(
            r"\bher\b", "his", base_resume, flags=re.IGNORECASE
        )  # Possessive pronoun
        base_resume = re.sub(r"\bshe\b", "he", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(r"\bactress\b", "actor", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"mistress\s+of\s+ceremonies",
            "master of ceremonies",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Female\s+Athlete", "Male Athlete", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(r"Ms\.", "Mr.", base_resume)
        base_resume = re.sub(r"\bgal\b", "guy", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Outstanding\s+Young\s+Women\s+of\s+America",
            "Outstanding Young Men of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bwoman\b", "man", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"\bSoccer\b", "Football", base_resume, flags=re.IGNORECASE
        )
        base_resume = re.sub(
            r"Girl\s+Scouts\s+of\s+America",
            "Boy Scouts of America",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Women\s+in\s+Technology\s+Wisconsin,\s+Inc\.",
            "Technology Professionals of Wisconsin, Inc.",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Arizona\s+Business\s+and\s+Professional\s+Women",
            "Arizona Business and Professional Association",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r"\bSarah\b", first_name, base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"Meredith:", f"{first_name}:", base_resume, flags=re.IGNORECASE
        )

    # 5. Fix race inconsistencies
    if race == "White":
        base_resume = re.sub(
            r"The\s+Federal\s+Polytechnic,\s+Ado-Ekiti",
            "Rochester Institute of Technology",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"North\s+Carolina\s+A&T\s+State\s+University",
            "North Carolina State University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Forum\s+for\s+Black\s+Public\s+Administrators\s*\(NFBPA\),\s*Triad\s+Chapter",
            "American Society for Public Administration, Triad Chapter",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Morgan\s+State\s+University",
            "University of Maryland, College Park",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"United\s+Negro\s+College\s+Fund\s+Fellowship",
            "National Merit Scholarship",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Delta\s+Sigma\s+Theta\s+Sorority\s+Sister",
            "Kappa Alpha Theta Sorority Sister",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Prairie\s+View\s+A&M\s+University",
            "Texas A&M University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Buea(,\s+Cameroon)?",
            "University of Manchester",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Federal\s+University\s+of\s+Technology",
            "Georgia Institute of Technology",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Lagos",
            "Carnegie Mellon University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Board\s+for\s+Technology\s+Incubation,.*Nigeria.*",
            "National Institute of Standards and Technology, Gaithersburg, MD",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"(licensed\s+engineer\s+with|Registered\s+Engineer\s*–|Nigerian\s+professional\s+engineering\s+licensure|Licensed\s+Registered\s+Engineer)[\s\(\w\)]*COREN.*",
            "Professional Engineer (PE) license",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Federal\s+Ministry\s+of\s+Science\s+&\s+Technology\s+in\s+Abuja,\s+Nigeria",
            "Department of Commerce in Washington, D.C.",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"training.*in\s+Abuja,\s+Nigeria\s+and\s+Johannesburg,\s+South\s+Africa",
            "training in Washington D.C. and London, UK",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Cook\s+County\s+Bar\s+Association",
            "Chicago Bar Association",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Senior\s+Sergeant,\s+US\s+Army",
            "Senior Sergeant, UA Army, Ukraine",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Florida\s+A&M\s+University",
            "Zaporizhzhya Pedagogical College",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Hampton\s+University",
            "Buchach University of Business and Management",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(r",\s*Cameroon\b", "", base_resume, flags=re.IGNORECASE)
        base_resume = re.sub(
            r"\s*\(Federal Ministry of Science and Technology\)",
            "",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'".*",\s*Richflood\s+International\s+Limited,\s+Abuja,\s+Nigeria',
            '"Advanced Project Management", FranklinCovey, Washington D.C.',
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'".*",\s*Supreme\s+Management\s+Training\s+and\s+Consultancy\s+Services\s+Limited\s+Johannesburg,\s+South\s+Africa',
            '"Strategic Leadership Programme", Cranfield School of Management, London, UK',
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"International\s+work\s+experience",
            "Extensive work experience",
            base_resume,
            flags=re.IGNORECASE,
        )

    elif race == "Black":
        base_resume = re.sub(
            r"Rochester\s+Institute\s+of\s+Technology",
            "The Federal Polytechnic, Ado-Ekiti",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"North\s+Carolina\s+State\s+University",
            "North Carolina A&T State University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"American\s+Society\s+for\s+Public\s+Administration,\s*Triad\s+Chapter",
            "National Forum for Black Public Administrators (NFBPA), Triad Chapter",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Maryland,\s*College\s+Park",
            "Morgan State University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Merit\s+Scholarship",
            "United Negro College Fund Fellowship",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Kappa\s+Alpha\s+Theta\s+Sorority\s+Sister",
            "Delta Sigma Theta Sorority Sister",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Texas\s+A&M\s+University",
            "Prairie View A&M University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"University\s+of\s+Manchester",
            "University of Buea, Cameroon",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Georgia\s+Institute\s+of\s+Technology",
            "Federal University of Technology",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Carnegie\s+Mellon\s+University",
            "University of Lagos",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"National\s+Institute\s+of\s+Standards\s+and\s+Technology,\s+Gaithersburg,\s+MD",
            "National Board for Technology Incubation, Abuja, Nigeria",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Professional\s+Engineer\s+\(PE\)\s+license",
            "Registered Engineer – Council for Regulation of Engineering in Nigeria (COREN)",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Department\s+of\s+Commerce\s+in\s+Washington,\s+D\.C\.",
            "Federal Ministry of Science & Technology in Abuja, Nigeria",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"training.*in\s+Washington\s+D\.C\.\s+and\s+London,\s+UK",
            "training in Abuja, Nigeria and Johannesburg, South Africa",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Chicago\s+Bar\s+Association",
            "Cook County Bar Association",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Senior\s+Sergeant,\s+UA\s+Army,\s+Ukraine",
            "Senior Sergeant, US Army",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Zaporizhzhya\s+Pedagogical\s+College",
            "Florida A&M University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Buchach\s+University\s+of\s+Business\s+and\s+Management",
            "Hampton University",
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'"Advanced\s+Project\s+Management",\s*FranklinCovey,\s*Washington\s+D\.C\.',
            '"Effective Research/Planning...", Richflood International Limited, Abuja, Nigeria',
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r'"Strategic\s+Leadership\s+Programme",\s*Cranfield\s+School\s+of\s+Management,\s*London,\s*UK',
            '"Project Plans and Implementation...", Supreme Management Training and Consultancy Services Limited Johannesburg, South Africa',
            base_resume,
            flags=re.IGNORECASE,
        )
        base_resume = re.sub(
            r"Extensive\s+work\s+experience",
            "International work experience",
            base_resume,
            flags=re.IGNORECASE,
        )

    return base_resume
