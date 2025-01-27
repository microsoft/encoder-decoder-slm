import logging
from dataclasses import dataclass
from pathlib import Path

from transformers import HfArgumentParser

from mu.models.modeling_mu import get_model
from mu.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


def _get_input_text():
    question = "What is the conflict the two brothers are fighting over?
    context = "Set in Spain, the play deals with a conflict between two brothers over their inheritance. Don Henrique is older than Don Jamie by a year; under the system of primogeniture, Henrique is the heir to their father's estate. The late father's will gives Jamie a small income, but Henrique treats his younger brother with rudeness and condescension, breeding a hostile relationship between the two. The problem is that Henrique and his wife, Violante, have been married for a dozen years but have no children—leaving Jamie as Henrique's heir.\nJamie is a member of a circle of aristocratic friends, which includes a boy named Ascanio. The boy is the son of poor parents, but is admired for his grace and nobility of character. Among Jamie's friends is Leandro, a lusty young man who is interested in the beautiful Amaranta. She is the wife of the rapacious lawyer Bartolus; the attorney keeps his wife closely watched, and Leandro has developed a scheme to seduce her. He masquerades as a wealthy law student come to take instruction from Bartolus. The go-between in this is Lopez, the local curate and the title character.\nDon Henrique, angered over Jamie's status as his heir, makes a radical move to change the situation: he files a legal suit (Bartolus is his lawyer) to have the boy Ascanio declared his heir. Henrique testifies that before he married Violante, he was engaged or \"precontracted\" to Ascanio's mother Jacinta, and that the boy is his natural son. (Like other plays of the era, The Spanish Curate exploits the legal and ethical ambiguity of the precontract, which in some interpretations was like a demi-marriage...but not quite.) After the child's birth, Henrique had second thoughts about the social gap between himself and Jacinta, and got the precontract cancelled. Jacinta can only affirm the basic truth of Henrique's testimony; and on that basis, Henrique wins his suit. Ascanio is now his legal heir, and Jamie is out.\nViolante, however, is outraged that Henrique has exposed this shameful affair and effectively thrown her infertility in her face. She bullies her husband into reversing course and driving Ascanio out of his house; Henrique offers the boy financial support, but the child returns to Jacinta and his pretended father Octavio. Violante is not satisfied with this, however; she reveals herself to be a truly ruthless person when she solicits Jamie to murder both Henrique and Ascanio and so come into his family fortune immediately.\nLeandro works his way into the trust of Bartolus, and tries to seduce Amaranta; she is tempted by him, but stands on her virtue and fidelity. When Bartolus finally becomes suspicious, Amaranta can show that she and Leandro have been in church, and not having a sexual assignation.\nThe plot comes to a head in the final act: Violante meets Jamie and his pretended accomplices for the double murder—only to have her plot exposed. Henrique is shocked into penitence by the exposure of his wife's murder plot—and reveals that he and Violante are not actually, fully legally, married after all. Bartolus too is cowed by his involvement in the matter, and vows to change his ways. Jamie has no problem accepting Ascanio as his nephew, now that their family relations are better ordered. Massinger ends the play with a couplet extolling the middle way in marital relations, between too much pliability in a husband (like Henrique) and too little (like Bartolus)—much like the concluding couplet in Massinger's later play The Picture."
    text = "Answer the following question based only on the context.\n###Question\n{}\n###Context\n{}\n###Answer\n".format(question, context)
    return text


@dataclass
class GenerateText2Text:
    model_path: Path = Path("artifacts/models/model.pt")
    device: str = "cuda"

    def __call__(self):
        tokenizer = get_tokenizer()
        model = get_model(self.model_path, device=self.device)

        input_text = _get_input_text()
        logger.info(f"Input text: {input_text}")
        tokenizer_output = tokenizer(input_text, return_tensors="pt").to(self.device)
        output = model.generate(
            input_ids=tokenizer_output["input_ids"],
            temperature=0.9,
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Output text: {output_text}")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser(GenerateText2Text)
    generate_text2text = parser.parse_args_into_dataclasses()[0]
    generate_text2text()


if __name__ == "__main__":
    main()
