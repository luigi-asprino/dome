package it.cnr.istc.stlab;

public class WeightedCategory {
	private String category;
	private double score;

	public WeightedCategory(String category, double score) {
		super();
		this.category = category;
		this.score = score;
	}

	public String getCategory() {
		return category;
	}

	public void setCategory(String category) {
		this.category = category;
	}

	public double getScore() {
		return score;
	}

	public void setScore(double score) {
		this.score = score;
	}

}
